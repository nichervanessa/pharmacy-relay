"""
Redeeming an activation code (main._op_activation_claim), against an
in-memory stand-in for Firestore.

One code is the account's licence for as many PCs as the owner sold it for:
the same PC may enter it again, a new PC only while seats are free, and a code
that cannot be counted stays strictly single-use.

    cd relay && python -m pytest tests
"""

import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402


# ── a small Firestore ─────────────────────────────────────────────────────────

class Snap:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class Ref:
    def __init__(self, store, path):
        self.store, self.path = store, path

    def get(self, transaction=None):
        return Snap(self.store.get(self.path))

    def set(self, data):
        self.store[self.path] = dict(data)

    def update(self, fields):
        doc = self.store.setdefault(self.path, {})
        for key, value in fields.items():          # "devices.pc1" → doc["devices"]["pc1"]
            *parents, last = key.split(".")
            node = doc
            for p in parents:
                node = node.setdefault(p, {})
            node[last] = value


class Collection:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def document(self, doc_id):
        return Ref(self.store, f"{self.name}/{doc_id}")


class Transaction:
    def update(self, ref, fields):
        ref.update(fields)


class Client:
    def __init__(self, store):
        self.store = store

    def collection(self, name):
        return Collection(self.store, name)

    def transaction(self):
        return Transaction()


class FakeFirestore:
    SERVER_TIMESTAMP = "now"

    def __init__(self):
        self.store = {}

    def client(self):
        return Client(self.store)

    @staticmethod
    def transactional(fn):
        return fn


class NoAuth:
    """No such login: the licence is still valid, admin simply is not granted."""
    @staticmethod
    def get_user_by_email(email):
        raise LookupError(email)


CODE = "ABCDEFGHJKLMNPQR"


@pytest.fixture
def fs(monkeypatch):
    fake = FakeFirestore()
    monkeypatch.setattr(main, "firestore", fake)
    monkeypatch.setattr(main, "fb_auth", NoAuth)
    return fake


def account(fs, max_devices=None, devices=None, **extra):
    doc = {"name": "Shifa Pharmacy", "devices": dict(devices or {}), **extra}
    if max_devices is not None:
        doc["maxDevices"] = max_devices
    fs.store["accounts/acc1"] = doc


def code(fs, **extra):
    fs.store[f"activationCodes/{CODE}"] = {"accountId": "acc1", "accountName": "Shifa Pharmacy",
                                           "adminEmail": "owner@shifa.iq", "exp": "", "used": False, **extra}


def redeem(pc):
    return main._op_activation_claim({"code": CODE, "device_id": pc, "device_name": pc.upper()}, None)


def refused(pc):
    with pytest.raises(HTTPException) as e:
        redeem(pc)
    return e.value.detail


PC1, PC2, PC3 = "pc" + "1" * 30, "pc" + "2" * 30, "pc" + "3" * 30


class TestOnePc:
    def test_the_first_pc_activates_and_takes_the_seat(self, fs):
        account(fs, max_devices=1)
        code(fs)

        assert redeem(PC1)["max_devices"] == 1
        assert list(fs.store["accounts/acc1"]["devices"]) == [PC1]
        assert fs.store[f"activationCodes/{CODE}"]["used"] is True

    def test_a_second_pc_is_refused(self, fs):
        account(fs, max_devices=1)
        code(fs)
        redeem(PC1)

        assert "Device limit reached" in refused(PC2)
        assert list(fs.store["accounts/acc1"]["devices"]) == [PC1]

    def test_the_same_pc_may_activate_again(self, fs):
        """Reinstalled, or its database restored: it already holds the seat."""
        account(fs, max_devices=1)
        code(fs)
        redeem(PC1)

        assert redeem(PC1)["account_id"] == "acc1"
        assert len(fs.store["accounts/acc1"]["devices"]) == 1


class TestSoldForSeveralPcs:
    def test_one_code_activates_as_many_pcs_as_were_bought(self, fs):
        account(fs, max_devices=2)
        code(fs)

        redeem(PC1)
        redeem(PC2)

        assert sorted(fs.store["accounts/acc1"]["devices"]) == [PC1, PC2]
        assert sorted(fs.store[f"activationCodes/{CODE}"]["pcs"]) == [PC1, PC2]
        assert "Device limit reached" in refused(PC3)

    def test_removing_a_pc_in_the_panel_frees_its_seat_for_another(self, fs):
        account(fs, max_devices=2)
        code(fs)
        redeem(PC1)
        redeem(PC2)
        del fs.store["accounts/acc1"]["devices"][PC1]

        redeem(PC3)

        assert sorted(fs.store["accounts/acc1"]["devices"]) == [PC2, PC3]

    def test_the_first_use_is_kept_as_it_was(self, fs):
        account(fs, max_devices=2)
        code(fs)
        redeem(PC1)
        redeem(PC2)

        assert fs.store[f"activationCodes/{CODE}"]["usedOn"] == PC1.upper()


class TestCodesThatCannotBeCounted:
    def test_an_account_with_no_number_of_pcs_keeps_the_code_single_use(self, fs):
        """Old accounts: not limited, so a reused code could run any number of PCs."""
        account(fs)                 # no maxDevices
        code(fs)
        redeem(PC1)

        assert "already been used" in refused(PC2)

    def test_an_old_build_that_names_no_pc_gets_one_use(self, fs):
        account(fs, max_devices=3)
        code(fs)
        main._op_activation_claim({"code": CODE}, None)

        with pytest.raises(HTTPException) as e:
            main._op_activation_claim({"code": CODE}, None)
        assert "already been used" in e.value.detail

    def test_a_code_with_no_account_is_single_use(self, fs):
        code(fs, accountId="")
        redeem(PC1)

        assert "already been used" in refused(PC2)


class TestStillRefused:
    def test_revoked(self, fs):
        account(fs, max_devices=2)
        code(fs, revoked=True)
        assert "revoked" in refused(PC1)

    def test_expired(self, fs):
        account(fs, max_devices=2)
        code(fs, exp="2020-01-01")
        assert "expired" in refused(PC1)

    def test_blocked_account(self, fs):
        account(fs, max_devices=2, status="blocked")
        code(fs)
        assert "blocked" in refused(PC1)

    def test_unknown_code(self, fs):
        assert "Invalid" in refused(PC1)
