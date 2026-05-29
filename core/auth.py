from __future__ import annotations

import os

import streamlit as st

FREE_PLAN_LIMIT = 3   # max ativos no plano gratuito

_DEV_USER_ID = "00000000-0000-0000-0000-000000000001"


# ── Supabase client ──────────────────────────────────────────────────────

def _sb():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


# ── Session helpers ──────────────────────────────────────────────────────

def get_user_id() -> str | None:
    return st.session_state.get("auth_user_id")


def get_email() -> str | None:
    return st.session_state.get("auth_email")


def get_plan() -> str:
    return st.session_state.get("auth_plan", "free")


def is_pro() -> bool:
    return get_plan() == "pro"


def logout() -> None:
    for k in [k for k in st.session_state if k.startswith("auth_")]:
        del st.session_state[k]
    st.cache_data.clear()
    st.rerun()


# ── Internal auth helpers ────────────────────────────────────────────────

def _set_session(user_id: str, email: str, plan: str) -> None:
    st.session_state["auth_user_id"] = user_id
    st.session_state["auth_email"] = email
    st.session_state["auth_plan"] = plan


def _fetch_plan(sb, user_id: str) -> str:
    try:
        r = sb.table("user_plans").select("plan").eq("user_id", user_id).execute()
        if r.data:
            return r.data[0]["plan"]
        sb.table("user_plans").upsert({"user_id": user_id, "plan": "free"}).execute()
    except Exception:
        pass
    return "free"


def _claim_legacy_data(sb, user_id: str) -> None:
    """Assign legacy JSONB data to first user if no data exists yet."""
    try:
        existing = sb.table("otimizador_assets").select("id").eq("user_id", user_id).limit(1).execute()
        if existing.data:
            return  # user already has data

        # No data for this user → import from legacy JSONB
        legacy = sb.table("otimizador_portfolio").select("assets").eq("id", 1).execute()
        if not legacy.data or not legacy.data[0].get("assets"):
            return

        for a in legacy.data[0]["assets"]:
            ticker = a.get("ticker", "")
            if not ticker:
                continue
            sb.table("otimizador_assets").upsert(
                {"user_id": user_id, "ticker": ticker, "name": "", "asset_type": "ETF", "exchange": ""},
                on_conflict="user_id,ticker",
            ).execute()
            sb.table("otimizador_allocations").upsert(
                {"user_id": user_id, "ticker": ticker, "target_pct": float(a.get("target_pct", 0))},
                on_conflict="user_id,ticker",
            ).execute()
            sb.table("otimizador_positions").upsert(
                {"user_id": user_id, "ticker": ticker, "quantity": float(a.get("quantity", 0))},
                on_conflict="user_id,ticker",
            ).execute()
    except Exception:
        pass


def _try_login(email: str, password: str) -> tuple[bool, str]:
    sb = _sb()
    if not sb:
        return False, "Supabase não configurado."
    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            plan = _fetch_plan(sb, res.user.id)
            _set_session(res.user.id, res.user.email or email, plan)
            return True, ""
        return False, "Credenciais inválidas."
    except Exception as e:
        msg = str(e)
        if "Invalid login credentials" in msg:
            return False, "Email ou password incorretos."
        if "Email not confirmed" in msg:
            return False, "Confirma o email antes de entrar (verifica a caixa de correio)."
        return False, msg


def _try_register(email: str, password: str) -> tuple[bool, str]:
    sb = _sb()
    if not sb:
        return False, "Supabase não configurado."
    try:
        res = sb.auth.sign_up({"email": email, "password": password})
        if res.user:
            sb.table("user_plans").upsert({"user_id": res.user.id, "plan": "free"}).execute()
            if res.session:
                plan = "free"
                _set_session(res.user.id, res.user.email or email, plan)
                _claim_legacy_data(sb, res.user.id)
                return True, ""
            # Email confirmation required
            return True, "confirm_email"
        return False, "Registo falhou — tenta novamente."
    except Exception as e:
        msg = str(e)
        if "already registered" in msg or "User already registered" in msg:
            return False, "Este email já está registado. Usa 'Entrar'."
        return False, msg


def _try_reset_password(email: str) -> tuple[bool, str]:
    sb = _sb()
    if not sb:
        return False, "Supabase não configurado."
    try:
        sb.auth.reset_password_for_email(email)
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Auth page (rendered when not authenticated) ──────────────────────────

def show_auth_page() -> bool:
    """Render login/register page. Returns True if user is authenticated."""

    # Already authenticated
    if st.session_state.get("auth_user_id"):
        return True

    # Local dev: no Supabase configured → bypass auth
    if not os.environ.get("SUPABASE_URL"):
        _set_session(_DEV_USER_ID, "dev@local", "pro")
        return True

    # Center the form
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("## 📊 Otimizador de Portfólio")
        st.caption("Gestão inteligente de portfólios de ativos")
        st.divider()

        tab_login, tab_reg = st.tabs(["Entrar", "Criar conta"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email_l = st.text_input("Email", placeholder="joao@exemplo.com")
                pass_l  = st.text_input("Password", type="password")
                ok_l    = st.form_submit_button("Entrar", type="primary",
                                                use_container_width=True)
            if ok_l:
                if not email_l or not pass_l:
                    st.error("Preenche email e password.")
                else:
                    with st.spinner("A autenticar…"):
                        success, err = _try_login(email_l, pass_l)
                    if success:
                        st.rerun()
                    else:
                        st.error(err)

            with st.expander("Esqueci a password"):
                with st.form("reset_form"):
                    email_r = st.text_input("Email para recuperação")
                    ok_r = st.form_submit_button("Enviar link", use_container_width=True)
                if ok_r and email_r:
                    ok, err = _try_reset_password(email_r)
                    if ok:
                        st.success("Email enviado — verifica a caixa de correio.")
                    else:
                        st.error(err)

        with tab_reg:
            with st.form("reg_form", clear_on_submit=False):
                email_n  = st.text_input("Email", key="reg_email")
                pass_n   = st.text_input("Password", type="password", key="reg_pass",
                                         help="Mínimo 6 caracteres")
                pass_n2  = st.text_input("Confirmar password", type="password",
                                         key="reg_pass2")
                ok_n     = st.form_submit_button("Criar conta", type="primary",
                                                 use_container_width=True)
            if ok_n:
                if not email_n or not pass_n:
                    st.error("Preenche todos os campos.")
                elif len(pass_n) < 6:
                    st.error("A password precisa de ter pelo menos 6 caracteres.")
                elif pass_n != pass_n2:
                    st.error("As passwords não coincidem.")
                else:
                    with st.spinner("A criar conta…"):
                        success, result = _try_register(email_n, pass_n)
                    if success and result == "confirm_email":
                        st.info("Conta criada! Verifica o email para confirmar antes de entrar.")
                    elif success:
                        st.rerun()
                    else:
                        st.error(result)

        st.divider()
        st.caption("Os teus dados são privados e só tu os podes ver.")

    return False
