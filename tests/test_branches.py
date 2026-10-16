"""
Branch reports (branch.report / branch.list), the install token that lets a
PC send one without a login, and stamping a pharmacy onto older logins.

    cd relay && python -m pytest tests
"""

import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402
from tests.test_activation import Client, Collection, FakeFirestore, Ref  # noqa: E402


class SubRef(Ref):
    def collection(self, name):
        return SubCollection(self.store, f"{self.path}/{name}")


class Doc:
    def __init__(self, doc_id, data):
        self.id, self._data = doc_id, data

    def to_dict(self):
        return dict(self._data)


class SubCollection(Collection):
    def document(self, doc_id):
        return SubRef(self.store, f"{self.name}/{doc_id}")

    def stream(self):
        prefix = self.name + "/"
        for path, data in list(self.store.items()):
            rest = path[len(prefix):]
            if path.startswith(prefix) and "/" not in rest:
                yield Doc(rest, data)


class SubClient(Client):
    def collection(self, name):
        return SubCollection(self.store, name)


class Firestore(FakeFirestore):
    def client(self):
        return SubClient(self.store)


class Auth:
    claims: dict = {}
    written: dict = {}

    class _User:
        def __init__(self, claims):
            self.custom_claims = claims

    @classmethod
    def get_user(cls, uid):
        return cls._User(dict(cls.claims.get(uid, {})))

    @classmethod
    def set_custom_user_claims(cls, uid, claims):
        cls.written[uid] = claims


PC = "pc" + "a" * 30
OTHER_PC = "pc" + "b" * 30


@pytest.fixture
def fs(monkeypatch):
    fake = Firestore()
    monkeypatch.setattr(main, "firestore", fake)
    Auth.claims, Auth.written = {}, {}
    monkeypatch.setattr(main, "fb_auth", Auth)
    fake.store["accounts/acc1"] = {"name": "Shifa", "devices": {}, "maxDevices": 3, "adminUid": "owner"}
    return fake


def admin(account="acc1", uid="owner"):
    return {"uid": uid, "role": "admin", "account": account}


def link(fs):
    out = main._op_device_register({"device_id": PC, "device_name": "Main", "want_install_token": True}, admin())
    return out["install_token"]


def report(token, device_id=PC, account="acc1", **data):
    return main._op_branch_report({"account": account, "device_id": device_id, "install_token": token,
                                   "branch_id": "db1", "data": {"name": "Erbil", **data}}, None)


class TestInstallToken:
    def test_a_sign_in_hands_the_pc_a_token_and_keeps_only_its_hash(self, fs):
        token = link(fs)
        kept = fs.store["accounts/acc1"]["installTokens"][PC]
        assert len(token) > 30 and token not in str(kept) and kept["hash"] == main._hash(token)

    def test_without_asking_there_is_none(self, fs):
        out = main._op_device_register({"device_id": PC, "device_name": "Main"}, admin())
        assert "install_token" not in out


class TestReports:
    def test_the_linked_pc_files_its_report(self, fs):
        report(link(fs), revenue=1000)
        doc = fs.store["accounts/acc1/branches/db1"]
        assert (doc["name"], doc["revenue"], doc["deviceId"]) == ("Erbil", 1000, PC)

    def test_a_wrong_token_writes_nothing(self, fs):
        link(fs)
        with pytest.raises(HTTPException) as e:
            report("guessed-token")
        assert e.value.status_code == 403
        assert "accounts/acc1/branches/db1" not in fs.store

    def test_a_token_from_another_pc_is_refused(self, fs):
        token = link(fs)
        with pytest.raises(HTTPException):
            report(token, device_id=OTHER_PC)

    def test_a_token_for_another_account_is_refused(self, fs):
        token = link(fs)
        fs.store["accounts/acc2"] = {"name": "Other", "devices": {}}
        with pytest.raises(HTTPException):
            report(token, account="acc2")

    def test_an_oversized_report_is_refused(self, fs):
        with pytest.raises(HTTPException) as e:
            report(link(fs), junk="x" * 50_000)
        assert e.value.status_code == 413


class TestList:
    def test_the_admin_reads_every_branch_of_their_account(self, fs):
        report(link(fs), revenue=1000)
        fs.store["accounts/acc2/branches/x"] = {"name": "Someone else's", "branchId": "x"}
        [b] = main._op_branch_list({}, admin())
        assert (b["branch_id"], b["name"], b["revenue"]) == ("db1", "Erbil", 1000)
        assert "deviceId" not in b

    def test_a_cashier_cannot(self, fs):
        with pytest.raises(HTTPException):
            main._op_branch_list({}, {"uid": "c1", "role": "cashier", "account": "acc1"})


class TestStamping:
    def test_an_older_login_gets_its_pharmacy_written_in(self, fs):
        Auth.claims["c1"] = {"role": "cashier", "created_by": "owner"}
        Auth.claims["owner"] = {"role": "admin", "account": "acc1"}
        main._op_device_register({"device_id": PC, "device_name": "Main"}, {"uid": "c1", "role": "cashier", "created_by": "owner"})
        assert Auth.written["c1"] == {"role": "cashier", "created_by": "owner", "account": "acc1"}

    def test_a_stamped_login_is_left_alone(self, fs):
        main._op_device_register({"device_id": PC, "device_name": "Main"}, admin())
        assert Auth.written == {}

    def test_account_of_answers_for_the_caller(self, fs):
        Auth.claims["owner"] = {"role": "admin", "account": "acc1"}
        assert main._op_account_of({}, {"uid": "c1", "role": "cashier", "created_by": "owner"}) == {"account": "acc1"}


class TestDashboardWrites:
    def _set(self, token, doc="owner", account="acc1"):
        return main._op_fs_set({"collection": "mobile_dashboard", "doc": doc, "data": {"x": 1},
                                "account": account, "device_id": PC, "install_token": token}, None)

    def test_a_linked_pc_writes_its_admins_dashboard(self, fs):
        self._set(link(fs))
        assert fs.store["mobile_dashboard/owner"] == {"x": 1}

    def test_a_wrong_token_writes_nothing(self, fs):
        link(fs)
        with pytest.raises(HTTPException):
            self._set("guessed")
        assert "mobile_dashboard/owner" not in fs.store

    def test_not_another_pharmacys_admin(self, fs):
        token = link(fs)
        Auth.claims["stranger"] = {"role": "admin", "account": "acc2"}
        with pytest.raises(HTTPException):
            self._set(token, doc="stranger")

    def test_the_secret_alone_can_be_closed(self, fs, monkeypatch):
        monkeypatch.setattr(main, "LEGACY_DASHBOARD_WRITES", False)
        with pytest.raises(HTTPException):
            main._op_fs_set({"collection": "mobile_dashboard", "doc": "owner", "data": {}}, None)
