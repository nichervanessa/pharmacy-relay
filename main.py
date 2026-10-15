"""
Firebase privilege relay
========================

A small hosted service that holds the Firebase service-account private key so
the desktop application never has to.

Why this exists
---------------
The desktop app needs a handful of privileged Firebase operations: creating
staff accounts, setting role claims, and reading/writing a few Firestore
documents. Performing those locally means shipping the service-account private
key inside every customer's installer — and anyone who extracts that key gains
total control of the Firebase project: minting themselves admin accounts,
forging activation codes, and reading every other pharmacy's data.

This relay keeps the key in one place you control. The desktop app calls it
over HTTPS and gets back only the specific result it asked for.

Protection layers
-----------------
1. Shared secret (X-Relay-Secret) — a coarse gate that keeps unrelated
   internet traffic out. It ships inside the app, so treat it as a filter and
   NOT as a credential. Nothing below may rely on it alone.
2. Caller identity — every operation that touches an account requires the
   caller's own Firebase ID token, verified here.
3. Entitlement — what the caller may then do is decided from data only the
   control panel can write (an `activationCodes` document, an `accounts`
   document), never from what the caller asks for.
4. Collection allow-list — Firestore access is limited to the exact
   collections this product uses, with per-collection rules.

The rule that has to hold
-------------------------
NO PATH MAY GRANT THE ADMIN ROLE ON THE STRENGTH OF THE SHARED SECRET, OR ON
THE STRENGTH OF THE CALLER SIMPLY BEING SIGNED IN.

An earlier version of this file broke that rule twice over. `user.set_claims`
accepted any caller setting claims on their OWN uid — so a cashier could name
themselves admin — and accepted a request with no caller at all as long as the
role being granted was "admin", on the strength of a comment saying the
desktop backend had checked an activation code first. The relay never verified
that, and could not: the check sat on the other side of a boundary the
attacker controls.

Both are gone. Becoming an admin now happens through exactly two operations,
`activation.claim` and `account.claim_admin`, and each decides FOR ITSELF,
from Firestore, both whether a grant is warranted and who receives it. The
caller supplies a code or a token — never a role, and never a target.

Deployment: see README.md.
"""

from __future__ import annotations

import datetime
import hmac
import logging
import os
from typing import Any, Optional

import firebase_admin
from firebase_admin import auth as fb_auth, credentials, firestore
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("relay")

RELAY_SECRET = (os.getenv("RELAY_SECRET") or "").strip()

# Rotating the shared secret invalidates every installer already in the field,
# because the secret is baked into the build. Accepting the previous one for a
# while makes a rotation something you can actually do: deploy the relay with
# both, ship the update, then clear this and deploy again. Leave it empty
# unless a rotation is in progress.
RELAY_SECRET_PREVIOUS = (os.getenv("RELAY_SECRET_PREVIOUS") or "").strip()

SERVICE_ACCOUNT = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT", "/etc/secrets/firebase_service_account.json"
)

# Firestore collections this product legitimately touches. Anything else is
# refused, so a compromised client cannot roam the database.
ALLOWED_COLLECTIONS = {"accounts", "activationCodes", "mobile_dashboard"}

# Claims a client may cause to be written. A custom claim rides in every ID
# token the account presents afterwards, so an unbounded dict here is both an
# escalation surface and a way to push a token past Firebase's 1000-byte claim
# limit and lock an account out of logging in entirely.
ALLOWED_CLAIM_KEYS = {"role", "created_by", "account"}
ALLOWED_ROLES = {"admin", "cashier", "pharmacist", "warehouse"}

app = FastAPI(title="Pharmacy Firebase Relay", version="2.0.0")


def _init() -> None:
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(SERVICE_ACCOUNT))


@app.on_event("startup")
def _startup() -> None:
    if not RELAY_SECRET:
        logger.warning("RELAY_SECRET is not set — every request will be refused.")
    if RELAY_SECRET_PREVIOUS:
        logger.warning(
            "RELAY_SECRET_PREVIOUS is set — a secret rotation is in progress. "
            "Clear it once the updated build has reached every customer."
        )
    _init()
    logger.info("Relay ready. Allowed collections: %s", sorted(ALLOWED_COLLECTIONS))


class OpIn(BaseModel):
    op: str
    args: dict = {}


# ─── Caller identity ──────────────────────────────────────────────────────────

def _caller(authorization: Optional[str]) -> Optional[dict]:
    """Verified claims of the calling staff member, or None if absent."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        return fb_auth.verify_id_token(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired caller token")


def _caller_uid(claims: dict) -> str:
    return str(claims.get("uid") or claims.get("user_id") or claims.get("sub") or "")


def _require_caller(claims: Optional[dict]) -> dict:
    if not claims:
        raise HTTPException(401, "This action requires a signed-in user")
    return claims


def _require_admin(claims: Optional[dict]) -> dict:
    _require_caller(claims)
    if claims.get("role") != "admin":
        raise HTTPException(403, "This action requires the admin role")
    return claims


def _check_collection(name: str) -> str:
    if name not in ALLOWED_COLLECTIONS:
        raise HTTPException(403, f"Collection '{name}' is not accessible through the relay")
    return name


def _clean_claims(raw: Any) -> dict:
    """Validate a claims dict down to the keys and values this product uses."""
    if not isinstance(raw, dict):
        raise HTTPException(400, "claims must be an object")
    unknown = set(raw) - ALLOWED_CLAIM_KEYS
    if unknown:
        raise HTTPException(400, f"Refusing to set unknown claims: {sorted(unknown)}")
    role = raw.get("role")
    if role not in ALLOWED_ROLES:
        raise HTTPException(400, f"Refusing to set unknown role '{role}'")
    cleaned = {"role": role}
    for key in ("created_by", "account"):
        value = raw.get(key)
        if value:
            if not isinstance(value, str) or len(value) > 128:
                raise HTTPException(400, f"Invalid value for claim '{key}'")
            cleaned[key] = value
    return cleaned


def _target_claims(uid: str) -> dict:
    """Existing claims of the account being acted on ({} if it has none)."""
    try:
        return fb_auth.get_user(uid).custom_claims or {}
    except Exception:
        raise HTTPException(404, "No such account")


def _may_manage(caller: dict, uid: str, target: dict) -> bool:
    """
    Whether an admin caller may act on this account.

    Every pharmacy in the customer base shares one Firebase project, so "is an
    admin" is not by itself a reason to let somebody edit an account — it has
    to be an admin of the SAME pharmacy. Ownership is read from claims the
    relay itself wrote: `created_by` (who created this staff member) and
    `account` (which panel account they belong to).

    Deliberately permissive in three places, each for a reason:
      • acting on oneself, which an admin re-stamping their own claims does;
      • an account with no role yet — the gap between user.create and the
        claim that gives it one;
      • an account with neither `created_by` nor `account`, which is what
        every staff member looked like before ownership tracking existed and
        is exactly what the claim-unowned migration is for.
    """
    if uid == _caller_uid(caller):
        return True
    if not target.get("role"):
        return True
    if not target.get("created_by") and not target.get("account"):
        return True
    if target.get("created_by") == _caller_uid(caller):
        return True
    caller_account = caller.get("account")
    return bool(caller_account and target.get("account") == caller_account)


def _require_may_manage(caller: dict, uid: str) -> dict:
    target = _target_claims(uid)
    if not _may_manage(caller, uid, target):
        raise HTTPException(403, "That account belongs to another pharmacy")
    return target


def _user_dict(u) -> dict:
    return {
        "uid": u.uid,
        "email": u.email or "",
        "display_name": u.display_name or "",
        "disabled": bool(u.disabled),
        "custom_claims": u.custom_claims or {},
        "phone_number": getattr(u, "phone_number", "") or "",
    }


# ─── Operations ───────────────────────────────────────────────────────────────
# Each handler takes (args, caller_claims) and returns a JSON-serialisable
# result. Handlers decide their own authorisation requirements.

def _op_user_list(args: dict, caller: Optional[dict]):
    _require_admin(caller)
    return [_user_dict(u) for u in fb_auth.list_users().users]


def _op_user_get(args: dict, caller: Optional[dict]):
    _require_admin(caller)
    return _user_dict(fb_auth.get_user(args["uid"]))


def _op_user_get_by_email(args: dict, caller: Optional[dict]):
    # Was reachable on the shared secret alone, which made the relay an email
    # directory for the whole customer base: ask about an address, learn
    # whether it exists and what role it holds. The activation flow was the
    # reason it had to be open; activation.claim below no longer needs it.
    _require_admin(caller)
    return _user_dict(fb_auth.get_user_by_email(args["email"]))


def _op_user_create(args: dict, caller: Optional[dict]):
    _require_admin(caller)
    kwargs = {
        "email": args["email"],
        "password": args["password"],
        "display_name": args.get("display_name", ""),
    }
    if args.get("phone_number"):
        kwargs["phone_number"] = args["phone_number"]
    return _user_dict(fb_auth.create_user(**kwargs))


def _op_user_update(args: dict, caller: Optional[dict]):
    _require_admin(caller)
    _require_may_manage(caller, args["uid"])
    fields = args.get("fields") or {}
    allowed = {"display_name", "password", "disabled", "email", "phone_number"}
    unknown = set(fields) - allowed
    if unknown:
        raise HTTPException(400, f"Cannot update: {sorted(unknown)}")
    return _user_dict(fb_auth.update_user(args["uid"], **fields))


def _op_user_delete(args: dict, caller: Optional[dict]):
    _require_admin(caller)
    _require_may_manage(caller, args["uid"])
    fb_auth.delete_user(args["uid"])
    return None


def _op_user_set_claims(args: dict, caller: Optional[dict]):
    """
    Set an account's role. ADMIN CALLER ONLY, and only within their own
    pharmacy.

    There is no bootstrap path through here any more, and that is the point.
    Becoming the first admin of an installation is not a claim change with a
    lenient rule in front of it; it is its own operation with its own proof —
    see activation.claim and account.claim_admin below.
    """
    _require_admin(caller)
    uid = args["uid"]
    claims = _clean_claims(args.get("claims") or {})
    _require_may_manage(caller, uid)
    fb_auth.set_custom_user_claims(uid, claims)
    return None


# ─── Becoming an admin ────────────────────────────────────────────────────────
# The only two ways. Both establish entitlement from a document the customer
# cannot write, and both choose the recipient themselves.

def _account_blocked(account_id: str) -> bool:
    if not account_id:
        return False
    try:
        snap = firestore.client().collection("accounts").document(account_id).get()
    except Exception:
        return False        # an unreadable account doc must not break activation
    return bool(snap.exists and (snap.to_dict() or {}).get("status") == "blocked")


# ─── PCs and sign-ins ─────────────────────────────────────────────────────────
# Two things the panel owner decides per account, and that only this side can
# hold a customer to, because every pharmacy PC keeps its own database and none
# of them can see another:
#
#   maxDevices       how many PCs the account may be installed on (the panel
#                    sets 1 on a new account; unset means not limited).
#                    Each PC that activates, or signs a member of staff in,
#                    takes a seat in accounts/{id}.devices. A PC with no seat
#                    is refused once the seats are full. The owner frees one by
#                    removing that PC in the panel.
#
#   allowMultiLogin  whether ONE login may be signed in on several PCs at the
#                    same time (default no). When it may not, the newest
#                    sign-in holds sessions/{uid} and the app on the other PC
#                    signs itself out.
#
# Both are answers the customer's PC acts on; neither grants anything.

DEVICE_LIMIT = "Device limit reached"


def _device_args(args: dict) -> tuple:
    """(device_id, device_name), or ("", "") from a build that sends none."""
    device_id = str(args.get("device_id") or "").strip().lower()
    if not device_id:
        return "", ""
    # Used as a Firestore field name inside `devices`, so it is held to what a
    # field path accepts unquoted: a letter, then letters and digits.
    if not (8 <= len(device_id) <= 64 and device_id[0].isalpha()
            and device_id.isascii() and device_id.isalnum()):
        raise HTTPException(400, "Invalid device id")
    return device_id, str(args.get("device_name") or "")[:60]


def _max_devices(account: dict) -> int:
    """
    How many PCs the account is licensed for; 0 when the owner has not said.

    "Not said" is every account created before PCs were counted, and some of
    those customers already run two tills. Holding them to one the day this
    shipped would have locked the second till out of its own shop, so an
    account is counted but not limited until the owner gives it a number —
    which the panel does for every new account, and on the next edit of an
    old one.
    """
    raw = account.get("maxDevices")
    if raw is None or raw == "":
        return 0
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _account_limits(account: dict) -> dict:
    def n(key: str, default: int) -> int:
        try:
            return max(0, int(account.get(key)))
        except (TypeError, ValueError):
            return default
    return {
        "admins": max(1, n("admins", 1)),
        "cashiers": n("cashiers", 0),
        "pharmacists": n("pharmacists", 0),
        "warehouse": n("warehouse", 0),
    }


def _seat(transaction, acc_ref, account: dict, device_id: str, device_name: str) -> None:
    """Keep this PC's seat on the account, give it one, or refuse. Writes only."""
    devices = account.get("devices") or {}
    if not isinstance(devices, dict):
        devices = {}
    if device_id in devices:
        transaction.update(acc_ref, {
            f"devices.{device_id}.lastSeen": firestore.SERVER_TIMESTAMP,
            f"devices.{device_id}.name": device_name,
        })
        return
    allowed = _max_devices(account)
    if allowed and len(devices) >= allowed:
        raise HTTPException(
            403,
            f"{DEVICE_LIMIT}: this account is licensed for {allowed} PC(s) and all "
            f"of them are in use. Contact the developer to add this PC.",
        )
    transaction.update(acc_ref, {f"devices.{device_id}": {
        "name": device_name,
        "firstSeen": firestore.SERVER_TIMESTAMP,
        "lastSeen": firestore.SERVER_TIMESTAMP,
    }})


def _account_of(caller: dict) -> str:
    """
    The panel account a signed-in member of staff belongs to; "" for a login no
    account owns (a self-hosted install, or one from before accounts existed).

    Read from claims this relay wrote: the admin carries `account`; staff carry
    it too on newer builds, and otherwise `created_by` names their admin.
    """
    account = str(caller.get("account") or "")
    if account:
        return account
    creator = str(caller.get("created_by") or "")
    if creator:
        try:
            account = str((fb_auth.get_user(creator).custom_claims or {}).get("account") or "")
        except Exception:
            account = ""
        if account:
            return account
    docs = list(firestore.client().collection("accounts")
                .where("adminUid", "==", _caller_uid(caller)).limit(1).get())
    return docs[0].id if docs else ""


def _op_device_register(args: dict, caller: Optional[dict]):
    """
    A member of staff has signed in on a PC. Seat the PC on their account (or
    refuse it), record this as their newest sign-in, and hand back what the
    panel currently says about the account — so a limit the owner changed
    reaches a PC that was activated long ago.

    Returns None for a login no panel account owns: nothing to enforce.
    """
    caller = _require_caller(caller)
    device_id, device_name = _device_args(args)
    if not device_id:
        raise HTTPException(400, "Missing argument: device_id")
    session_id = str(args.get("session_id") or "")[:80]

    account_id = _account_of(caller)
    if not account_id:
        return None

    client = firestore.client()
    acc_ref = client.collection("accounts").document(account_id)

    @firestore.transactional
    def _run(transaction):
        snap = acc_ref.get(transaction=transaction)
        if not snap.exists:
            return None
        account = snap.to_dict() or {}
        if account.get("status") == "blocked":
            raise HTTPException(403, "This account has been blocked. Contact the developer.")
        _seat(transaction, acc_ref, account, device_id, device_name)
        return account

    account = _run(client.transaction())
    if account is None:
        return None

    if session_id:
        client.collection("sessions").document(_caller_uid(caller)).set({
            "deviceId": device_id,
            "deviceName": device_name,
            "sessionId": session_id,
            "accountId": account_id,
            "at": firestore.SERVER_TIMESTAMP,
        })

    return {
        "account": account_id,
        "max_devices": _max_devices(account),
        "allow_multi_login": bool(account.get("allowMultiLogin")),
        "limits": _account_limits(account),
    }


def _op_session_active(args: dict, caller: Optional[dict]):
    """Where the CALLER's newest sign-in is: {device_id, session_id}, or None."""
    caller = _require_caller(caller)
    snap = firestore.client().collection("sessions").document(_caller_uid(caller)).get()
    if not snap.exists:
        return None
    data = snap.to_dict() or {}
    return {
        "device_id": str(data.get("deviceId") or ""),
        "session_id": str(data.get("sessionId") or ""),
        "device_name": str(data.get("deviceName") or ""),
    }


def _op_activation_claim(args: dict, caller: Optional[dict]):
    """
    Redeem an activation code: verify it, burn it, and grant admin to the
    account the code names — all here, in one operation.

    The desktop app used to do this in three separate calls (read the code,
    write the claim, mark the code used), which meant the middle one had to be
    permitted to a caller who was not yet an admin. Possession of an unused
    code is the credential; splitting the check from the grant left the relay
    trusting a client's word that it had seen one.

    The code document decides who becomes admin. A caller cannot pass a uid.

    Returns the licence details the desktop app stores locally.
    """
    code = str(args.get("code") or "").strip().upper()
    if not code or not code.isalnum() or len(code) != 16:
        raise HTTPException(400, "Invalid activation code")

    # The PC redeeming it. An older build sends none and is activated as it
    # always was — it simply takes no seat until it signs somebody in.
    device_id, device_name = _device_args(args)

    client = firestore.client()
    code_ref = client.collection("activationCodes").document(code)
    seat = {}

    # A transaction, because two tills redeeming the same code at the same
    # moment must not both succeed. Read-check-write inside one makes the
    # single-use rule real rather than merely likely.
    @firestore.transactional
    def _burn(transaction):
        snap = code_ref.get(transaction=transaction)
        if not snap.exists:
            raise HTTPException(400, "Invalid activation code")
        data = snap.to_dict() or {}
        if data.get("revoked"):
            raise HTTPException(400, "This activation code has been revoked")
        already_used = bool(data.get("used"))

        exp = str(data.get("exp") or "")
        if exp:
            try:
                if datetime.date.fromisoformat(exp) < datetime.date.today():
                    raise HTTPException(400, "This activation code has expired")
            except ValueError:
                pass        # an unparseable expiry is treated as no expiry

        # The account is read INSIDE the transaction: the seat count must be
        # the one this redemption is judged against, and a refusal has to come
        # before the code is spent — a customer told "no room for this PC"
        # must still be holding a code that works once room is made.
        account_id = str(data.get("accountId") or "")
        acc_ref = client.collection("accounts").document(account_id) if account_id else None
        account = None
        if acc_ref is not None:
            try:
                acc_snap = acc_ref.get(transaction=transaction)
                account = (acc_snap.to_dict() or {}) if acc_snap.exists else None
            except Exception:
                account = None      # an unreadable account doc must not break activation

        if account and account.get("status") == "blocked":
            raise HTTPException(403, "This account has been blocked. Contact the developer.")

        # One code is the account's licence for as many PCs as the owner sold
        # it for (maxDevices: 1, 2, 3 …). Once spent it may be entered again:
        #   • on a PC that already holds one of the account's seats — the same
        #     PC reinstalled, or its database restored — and
        #   • on a new PC while the account still has a free seat (_seat
        #     below refuses it, "Device limit reached", once they are all taken).
        # Never for a code without an account, an old build that sends no PC,
        # or an account the owner has not given a number of PCs: none of
        # those can be counted, so the code stays strictly single-use.
        if already_used and not (account is not None and device_id and _max_devices(account)):
            raise HTTPException(400, "This activation code has already been used")

        if account is not None:
            seat["max_devices"] = _max_devices(account)
            seat["allow_multi_login"] = bool(account.get("allowMultiLogin"))
            if device_id:
                _seat(transaction, acc_ref, account, device_id, device_name)

        spent = {f"pcs.{device_id}": {"name": device_name, "at": firestore.SERVER_TIMESTAMP}} if device_id else {}
        if not already_used:
            spent.update({
                "used": True,
                "usedBy": data.get("adminEmail") or "",
                "usedOn": device_name,
                "usedAt": firestore.SERVER_TIMESTAMP,
            })
        transaction.update(code_ref, spent)
        return data

    data = _burn(client.transaction())

    account_id = str(data.get("accountId") or "")
    admin_email = str(data.get("adminEmail") or "").strip().lower()

    # Grant admin to the account the CODE names — not to whoever called.
    admin_granted = False
    if admin_email:
        try:
            target = fb_auth.get_user_by_email(admin_email)
            claims = dict(target.custom_claims or {})
            claims.update({"role": "admin", "account": account_id})
            fb_auth.set_custom_user_claims(target.uid, _clean_claims(claims))
            admin_granted = True
        except HTTPException:
            raise
        except Exception as e:
            # The licence is still valid; the login may simply not exist yet.
            logger.warning("activation %s: could not grant admin to %s: %s",
                           code, admin_email, e)

    return {
        "account_id": account_id,
        "account_name": str(data.get("accountName") or ""),
        "exp": str(data.get("exp") or ""),
        "limits": data.get("limits") or {},
        "admin_email": admin_email,
        "admin_granted": admin_granted,
        "max_devices": seat.get("max_devices", 0),
        "allow_multi_login": seat.get("allow_multi_login", False),
    }


def _op_account_claim_admin(args: dict, caller: Optional[dict]):
    """
    Grant admin to the CALLER, if the control panel created them as the admin
    of a customer account.

    Identity comes from the caller's verified ID token; entitlement comes from
    an `accounts` document only the panel owner can write. The caller supplies
    neither a uid nor a role, so there is nothing here to aim at somebody else.
    """
    caller = _require_caller(caller)
    uid = _caller_uid(caller)
    email = str(caller.get("email") or "").strip().lower()

    collection = firestore.client().collection("accounts")
    docs = list(collection.where("adminUid", "==", uid).limit(1).get())
    if not docs and email and caller.get("email_verified"):
        # Matching on the email address only when Firebase has verified it.
        # Otherwise anyone could sign up claiming a customer's address and
        # inherit their pharmacy.
        docs = list(collection.where("adminEmail", "==", email).limit(1).get())

    if not docs:
        return None

    doc = docs[0]
    data = doc.to_dict() or {}
    if data.get("status") == "blocked":
        raise HTTPException(403, "This account has been blocked. Contact the developer.")

    claims = dict(fb_auth.get_user(uid).custom_claims or {})
    claims.update({"role": "admin", "account": doc.id})
    fb_auth.set_custom_user_claims(uid, _clean_claims(claims))

    # Record the uid so the next login matches on it directly.
    if not data.get("adminUid"):
        try:
            collection.document(doc.id).update({"adminUid": uid})
        except Exception:
            pass

    return {"account": doc.id, "account_name": str(data.get("accountName") or "")}


# ─── Firestore ────────────────────────────────────────────────────────────────

def _op_fs_get(args: dict, caller: Optional[dict]):
    # Reading licence documents used to be open on the shared secret, because
    # that was how activation worked: fetch the code, decide locally. It also
    # let anyone holding the secret probe for valid codes at their leisure.
    # activation.claim replaced that flow, and nothing in the desktop app calls
    # this any more — so it takes an admin, like every other read here.
    _require_admin(caller)
    coll = _check_collection(args["collection"])
    snap = firestore.client().collection(coll).document(args["doc"]).get()
    return snap.to_dict() if snap.exists else None


def _op_fs_query(args: dict, caller: Optional[dict]):
    _require_admin(caller)
    coll = _check_collection(args["collection"])
    docs = (
        firestore.client()
        .collection(coll)
        .where(args["field"], "==", args["value"])
        .limit(int(args.get("limit") or 1))
        .get()
    )
    return [{"id": d.id, "data": d.to_dict() or {}} for d in docs]


def _op_fs_update(args: dict, caller: Optional[dict]):
    coll = _check_collection(args["collection"])
    if coll == "activationCodes":
        # Burning a code is what activation.claim does — transactionally, and
        # together with the grant it pays for. There is no longer any reason
        # for a client to write one of these documents directly.
        raise HTTPException(403, "Activation codes are written by activation.claim only")
    _require_admin(caller)

    data = dict(args.get("data") or {})
    for field in args.get("server_timestamp_fields") or []:
        data[field] = firestore.SERVER_TIMESTAMP
    firestore.client().collection(coll).document(args["doc"]).update(data)
    return None


def _op_fs_set(args: dict, caller: Optional[dict]):
    """
    Write an admin's mobile-dashboard snapshot.

    ── The one operation still reachable on the shared secret alone ──────────
    Deliberately, and with its limits written down rather than assumed.

    This is pushed by the desktop's background scheduler, not by a request, so
    there is no signed-in user to authenticate as: the sync runs on a timer
    whether anyone is at the till or not, and an ID token expires in an hour.
    Requiring a caller here would simply stop the phone dashboard updating.

    What that costs is bounded and unchanged from before this rework: somebody
    holding the secret can overwrite a `mobile_dashboard/{uid}` document, which
    makes an owner's phone show wrong figures until the next sync. It reads
    nothing, reaches no other collection, and grants no role.

    When a caller IS present the rule tightens to "your own document only", so
    the manual Sync-now button is held to the stricter standard.

    To close it properly, mint a per-install token during activation.claim and
    require it here — see README.md.
    """
    coll = _check_collection(args["collection"])
    if coll != "mobile_dashboard":
        # Only the dashboard snapshot is written wholesale; account and code
        # documents are created by the control panel, never by a customer.
        raise HTTPException(403, f"Cannot create documents in '{coll}'")

    doc = str(args["doc"] or "")
    # Firebase uids are short opaque strings. Checking the shape stops a
    # malformed or path-like id from reaching Firestore at all.
    if not doc or len(doc) > 128 or "/" in doc:
        raise HTTPException(400, "Invalid dashboard document id")

    if caller and doc != _caller_uid(caller):
        raise HTTPException(403, "You can only write your own dashboard")

    firestore.client().collection(coll).document(doc).set(args.get("data") or {})
    return None


HANDLERS = {
    "user.list": _op_user_list,
    "user.get": _op_user_get,
    "user.get_by_email": _op_user_get_by_email,
    "user.create": _op_user_create,
    "user.update": _op_user_update,
    "user.delete": _op_user_delete,
    "user.set_claims": _op_user_set_claims,
    "activation.claim": _op_activation_claim,
    "account.claim_admin": _op_account_claim_admin,
    "device.register": _op_device_register,
    "session.active": _op_session_active,
    "fs.get": _op_fs_get,
    "fs.query": _op_fs_query,
    "fs.update": _op_fs_update,
    "fs.set": _op_fs_set,
}


# ─── Entry point ──────────────────────────────────────────────────────────────

def _secret_ok(presented: Optional[str]) -> bool:
    """Constant-time check against the current secret, and the previous one
    while a rotation is in flight."""
    if not RELAY_SECRET:
        return False
    presented = presented or ""
    if hmac.compare_digest(presented, RELAY_SECRET):
        return True
    if RELAY_SECRET_PREVIOUS and hmac.compare_digest(presented, RELAY_SECRET_PREVIOUS):
        logger.info("request authenticated with the PREVIOUS relay secret")
        return True
    return False


@app.post("/op")
def run_op(
    payload: OpIn,
    x_relay_secret: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    if not _secret_ok(x_relay_secret):
        raise HTTPException(401, "Unauthorised")

    handler = HANDLERS.get(payload.op)
    if handler is None:
        raise HTTPException(400, f"Unknown operation '{payload.op}'")

    _init()
    caller = _caller(authorization)

    try:
        result = handler(payload.args or {}, caller)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(400, f"Missing argument: {e}")
    except Exception as e:
        # Logged in full here, summarised to the caller. The old code returned
        # str(e) verbatim, which handed Firebase's internal error text — and
        # on occasion document contents — to anyone holding the shared secret.
        logger.warning("op %s failed: %s", payload.op, e, exc_info=True)
        raise HTTPException(400, f"The operation '{payload.op}' could not be completed")

    return {"result": result}


@app.get("/health")
def health():
    return {"status": "healthy", "configured": bool(RELAY_SECRET)}
