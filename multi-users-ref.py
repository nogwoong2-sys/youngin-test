"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇
실행: streamlit run 7.MultiService/code/multi-users-ref.py
필수 환경변수(Secrets/.env): SUPABASE_URL, SUPABASE_ANON_KEY
"""

from __future__ import annotations

import json
import importlib
import logging
import os
import re
import secrets
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, cast

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
LOG_DIR = PROJECT_ROOT / "logs"
LOGO_PATH = PROJECT_ROOT / "영인로고.png"

SYSTEM_PROMPT = (
    "너는 친절한 한국어 어시스턴트다. 참고 문맥을 우선 활용하고, 모르면 모른다고 말한다. "
    "답변은 마크다운 헤딩(# ## ###)으로 구조화하고 존대말을 쓴다. "
    "구분선(---, ===, ___)과 취소선(~~)은 쓰지 않는다."
)

FOLLOWUP_PROMPT = """사용자 질문과 답변을 바탕으로, 이어서 물어보면 좋은 질문 3개를 한국어로만 제시하라.
출력 형식(이 형식만 사용):
### 💡 다음에 물어볼 수 있는 질문들
1. ...
2. ...
3. ...
"""


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root.handlers.clear()
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"chatbot_{datetime.now():%Y%m%d}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(fh)
    except (PermissionError, OSError):
        # Streamlit Cloud 등 쓰기 제한 환경에서는 파일 로깅을 생략한다.
        root.addHandler(logging.NullHandler())


def remove_separators(text: str) -> str:
    text = re.sub(r"~~[^~]+~~", "", text)
    text = re.sub(r"^[\t ]*([\-_=])\1{2,}[\t ]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def embedding_to_pgvector_str(values: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


def _as_rows(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    return [cast(dict[str, Any], row) for row in data if isinstance(row, dict)]


def _first_id_or_empty(data: Any) -> str:
    rows = _as_rows(data)
    if not rows:
        return ""
    return str(rows[0].get("id", ""))


def app_users_schema_issue(supabase: Client) -> str:
    try:
        res = (
            supabase.table("app_users")
            .select("id,login_id,password_hash,password_salt")
            .limit(1)
            .execute()
        )
        _ = res.data
        return ""
    except Exception as e:
        return str(e)


def normalize_login_id(login_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "", login_id.strip())


def make_password_hash(password: str, salt_hex: str) -> str:
    raw = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    )
    return raw.hex()


def find_app_user_by_login_id(supabase: Client, login_id: str) -> dict[str, Any] | None:
    res = (
        supabase.table("app_users")
        .select("id,login_id,password_hash,password_salt")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = _as_rows(res.data)
    return rows[0] if rows else None


def create_app_user(supabase: Client, login_id: str, password: str) -> str:
    salt_hex = secrets.token_hex(16)
    pw_hash = make_password_hash(password, salt_hex)
    payload = {
        "login_id": login_id,
        "password_hash": pw_hash,
        "password_salt": salt_hex,
    }
    try:
        created = supabase.table("app_users").insert(payload).execute()
        return _first_id_or_empty(created.data)
    except Exception as e:
        msg = str(e)
        # Backward compatibility:
        # old schema may still have auth_user_id NOT NULL.
        if "auth_user_id" in msg and "not-null" in msg.lower():
            payload_with_legacy = {
                **payload,
                "auth_user_id": str(uuid.uuid4()),
            }
            created = supabase.table("app_users").insert(payload_with_legacy).execute()
            return _first_id_or_empty(created.data)
        raise


def verify_app_user_password(user_row: dict[str, Any], password: str) -> bool:
    salt = str(user_row.get("password_salt", ""))
    saved = str(user_row.get("password_hash", ""))
    if not salt or not saved:
        return False
    calc = make_password_hash(password, salt)
    return secrets.compare_digest(saved, calc)


def fetch_sessions(supabase: Client, app_user_id: str) -> list[dict[str, Any]]:
    res = (
        supabase.table("chat_sessions")
        .select("id,title,updated_at")
        .eq("app_user_id", app_user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return _as_rows(res.data)


def fetch_messages(supabase: Client, session_id: str) -> list[dict[str, str]]:
    res = (
        supabase.table("chat_messages")
        .select("role,content,sort_order")
        .eq("session_id", session_id)
        .order("sort_order")
        .execute()
    )
    rows = _as_rows(res.data)
    return [
        {"role": str(r.get("role", "")), "content": str(r.get("content", ""))}
        for r in rows
        if r.get("role") in ("user", "assistant")
    ]


def replace_session_messages(
    supabase: Client, session_id: str, messages: list[dict[str, str]]
) -> None:
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    if not messages:
        supabase.table("chat_sessions").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", session_id).execute()
        return
    payload = [
        {
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
            "sort_order": i,
        }
        for i, m in enumerate(messages)
    ]
    supabase.table("chat_messages").insert(payload).execute()
    supabase.table("chat_sessions").update(
        {"updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def create_chat_session(supabase: Client, app_user_id: str, title: str) -> str:
    res = (
        supabase.table("chat_sessions")
        .insert({"title": title, "app_user_id": app_user_id})
        .execute()
    )
    return _first_id_or_empty(res.data)


def update_session_title(supabase: Client, session_id: str, title: str) -> None:
    supabase.table("chat_sessions").update(
        {"title": title, "updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def delete_chat_session(supabase: Client, session_id: str) -> None:
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()


def list_vector_filenames(supabase: Client, session_id: str) -> list[str]:
    res = (
        supabase.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    rows = _as_rows(res.data)
    return sorted({str(r.get("file_name")) for r in rows if r.get("file_name")})


def copy_vectors_between_sessions(
    supabase: Client, from_session_id: str, to_session_id: str, batch_size: int = 10
) -> None:
    res = (
        supabase.table("vector_documents")
        .select("content,file_name,embedding,metadata")
        .eq("session_id", from_session_id)
        .execute()
    )
    rows = _as_rows(res.data)
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        ins: list[dict[str, Any]] = []
        for r in batch:
            emb = r.get("embedding")
            if isinstance(emb, list):
                emb = embedding_to_pgvector_str([float(x) for x in emb])
            ins.append(
                {
                    "session_id": to_session_id,
                    "content": str(r.get("content", "")),
                    "file_name": str(r.get("file_name", "")),
                    "embedding": emb,
                    "metadata": r.get("metadata") or {},
                }
            )
        if ins:
            supabase.table("vector_documents").insert(ins).execute()


def insert_pdf_chunks(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    file_name: str,
    chunks: list[Document],
    batch_size: int = 10,
) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        vecs = embeddings.embed_documents(texts)
        rows = []
        for doc, vec in zip(batch, vecs, strict=True):
            meta = dict(doc.metadata or {})
            meta.setdefault("source", file_name)
            rows.append(
                {
                    "session_id": session_id,
                    "content": doc.page_content,
                    "file_name": file_name,
                    "embedding": embedding_to_pgvector_str(vec),
                    "metadata": meta,
                }
            )
        supabase.table("vector_documents").insert(rows).execute()


def retrieve_with_rpc(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    app_user_id: str,
    query: str,
    k: int = 6,
) -> list[Document]:
    qvec = embeddings.embed_query(query)
    try:
        res = supabase.rpc(
            "match_vector_documents",
            {
                "query_embedding": qvec,
                "match_count": k,
                "filter_session_id": session_id,
                "filter_app_user_id": app_user_id,
            },
        ).execute()
    except Exception:
        try:
            res = supabase.rpc(
                "match_vector_documents",
                {
                    "query_embedding": embedding_to_pgvector_str(qvec),
                    "match_count": k,
                    "filter_session_id": session_id,
                    "filter_app_user_id": app_user_id,
                },
            ).execute()
        except Exception:
            return _retrieve_fallback_scan(supabase, embeddings, session_id, query, k)
    out: list[Document] = []
    for row in _as_rows(res.data):
        out.append(
            Document(
                page_content=str(row.get("content", "")),
                metadata={
                    "file_name": row.get("file_name", ""),
                    "id": row.get("id"),
                    **(row.get("metadata") or {}),
                },
            )
        )
    return out


def _retrieve_fallback_scan(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    query: str,
    k: int,
) -> list[Document]:
    res = (
        supabase.table("vector_documents")
        .select("id,content,file_name,metadata,embedding")
        .eq("session_id", session_id)
        .limit(300)
        .execute()
    )
    rows = _as_rows(res.data)
    if not rows:
        return []
    q = embeddings.embed_query(query)

    def parse_emb(val: Any) -> list[float] | None:
        if isinstance(val, list):
            return [float(x) for x in val]
        if isinstance(val, str) and val.startswith("["):
            try:
                loaded = json.loads(val)
                if isinstance(loaded, list):
                    return [float(x) for x in loaded]
                return None
            except json.JSONDecodeError:
                return None
        return None

    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        ev = parse_emb(r.get("embedding"))
        if not ev or len(ev) != len(q):
            continue
        dot = sum(a * b for a, b in zip(q, ev, strict=True))
        na = sum(a * a for a in q) ** 0.5
        nb = sum(b * b for b in ev) ** 0.5
        sim = dot / (na * nb) if na and nb else 0.0
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    docs: list[Document] = []
    for _, r in scored[:k]:
        docs.append(
            Document(
                page_content=r["content"],
                metadata={"file_name": r.get("file_name", ""), **(r.get("metadata") or {})},
            )
        )
    return docs


def to_lc_messages(history: list[dict[str, str]]) -> list[Any]:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs


def make_llm(provider: str, model_name: str, temp: float, stream: bool) -> Any:
    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temp, streaming=stream)
    if provider == "anthropic":
        mod = importlib.import_module("langchain_anthropic")
        ChatAnthropic = getattr(mod, "ChatAnthropic")
        return ChatAnthropic(model=model_name, temperature=temp, streaming=stream)
    mod = importlib.import_module("langchain_google_genai")
    ChatGoogleGenerativeAI = getattr(mod, "ChatGoogleGenerativeAI")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temp)


def llm_invoke_text(llm: Any, prompt: Any) -> str:
    out = llm.invoke(prompt)
    return (out.content or "").strip() if hasattr(out, "content") else str(out).strip()


def generate_session_title(llm: Any, messages: list[dict[str, str]]) -> str:
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    first_asst = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    if not first_user and not first_asst:
        return "새 대화"
    prompt = (
        "다음은 채팅의 첫 사용자 질문과 첫 답변 일부다. "
        "이 대화를 대표하는 짧은 한국어 세션 제목(40자 이내, 따옴표 없이)만 출력하라.\n\n"
        f"[질문]\n{first_user[:800]}\n\n[답변 일부]\n{first_asst[:800]}"
    )
    try:
        title = llm_invoke_text(llm, prompt).split("\n")[0].strip()
        title = re.sub(r'^["\']|["\']$', "", title)
        return title[:80] if title else "저장된 대화"
    except Exception:
        return (first_user[:40] + "…") if len(first_user) > 40 else (first_user or "저장된 대화")


def generate_followup_block(llm: Any, question: str, answer: str, context: str) -> str:
    human = (
        f"{FOLLOWUP_PROMPT}\n\n[질문]\n{question}\n\n[답변]\n{answer[:4000]}\n\n"
        f"[참고문맥 일부]\n{context[:2000]}"
    )
    try:
        return remove_separators(llm_invoke_text(llm, human))
    except Exception:
        return (
            "### 💡 다음에 물어볼 수 있는 질문들\n"
            "1. 문서에서 핵심 정의를 더 자세히 설명해 달라고 요청하기\n"
            "2. 이 내용과 관련된 예시를 추가로 알려 달라고 하기\n"
            "3. 요약하거나 표로 정리해 달라고 하기"
        )


def process_uploaded_pdfs(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    uploaded_files: list[Any],
) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for uf in uploaded_files:
        raw = uf.getvalue()
        suffix = Path(uf.name).suffix or ".pdf"
        safe_name = Path(uf.name).name
        tmp = PROJECT_ROOT / f"_tmp_{uuid.uuid4().hex}{suffix}"
        try:
            tmp.write_bytes(raw)
            docs = PyPDFLoader(str(tmp)).load()
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata = dict(c.metadata or {})
            c.metadata["file_name"] = safe_name
        insert_pdf_chunks(supabase, embeddings, session_id, safe_name, chunks, batch_size=10)
    supabase.table("chat_sessions").update(
        {"updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def render_header() -> None:
    st.markdown(
        """
<style>
    h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
    h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
    h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
    div.stButton > button:first-child {
        background-color: #ff69b4;
        color: #ffffff;
    }
</style>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<div style="text-align:center;">
  <span style="font-size:3rem !important; font-weight:700;">
    <span style="color:#1f77b4 !important;">PDF 기반</span>
    <span style="color:#ffd700 !important;"> 멀티유저 멀티세션 RAG 챗봇</span>
  </span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def ensure_bootstrap(supabase: Client, app_user_id: str) -> None:
    sessions = fetch_sessions(supabase, app_user_id)
    if not sessions:
        sid = create_chat_session(supabase, app_user_id, "새 대화")
        st.session_state.current_session_id = sid
        st.session_state.messages = []
        return
    sid = str(sessions[0]["id"])
    st.session_state.current_session_id = sid
    st.session_state.messages = fetch_messages(supabase, sid)


def auth_panel(supabase: Client) -> tuple[bool, str]:
    with st.sidebar:
        st.markdown("### 사용자 인증")
        login_id = st.text_input("Login ID", key="login_id_input")
        password = st.text_input("Password", type="password", key="password_input")
        clean_login_id = normalize_login_id(login_id)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("회원가입", use_container_width=True):
                if not clean_login_id or len(password) < 6:
                    st.error("Login ID를 입력하고 비밀번호는 6자 이상으로 설정하세요.")
                else:
                    try:
                        exists = find_app_user_by_login_id(supabase, clean_login_id)
                        if exists:
                            st.error("이미 존재하는 Login ID입니다.")
                        else:
                            app_user_id = create_app_user(supabase, clean_login_id, password)
                            st.session_state.app_user_id = app_user_id
                            st.session_state.login_id = clean_login_id
                            ensure_bootstrap(supabase, app_user_id)
                            st.success("회원가입 및 로그인 완료")
                            st.rerun()
                    except Exception as e:
                        st.error(f"회원가입 실패: {e}")
        with c2:
            if st.button("로그인", use_container_width=True):
                if not clean_login_id or not password:
                    st.error("Login ID와 Password를 입력하세요.")
                else:
                    try:
                        user = find_app_user_by_login_id(supabase, clean_login_id)
                        if not user:
                            st.error("로그인 실패: 존재하지 않는 Login ID입니다.")
                        elif not verify_app_user_password(user, password):
                            st.error("로그인 실패: 비밀번호가 올바르지 않습니다.")
                        else:
                            app_user_id = str(user.get("id", ""))
                            st.session_state.app_user_id = app_user_id
                            st.session_state.login_id = clean_login_id
                            ensure_bootstrap(supabase, app_user_id)
                            st.success("로그인 성공")
                            st.rerun()
                    except Exception as e:
                        st.error(f"로그인 실패: {e}")

        logged_in = bool(st.session_state.get("app_user_id"))
        if logged_in:
            st.caption(f"현재 사용자: {st.session_state.get('login_id', '')}")
            if st.button("로그아웃", use_container_width=True):
                for k in ["app_user_id", "login_id", "messages", "current_session_id"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        return logged_in, st.session_state.get("app_user_id", "")


def main() -> None:
    setup_logging()
    st.set_page_config(
        page_title="PDF 기반 멀티유저 멀티세션 RAG 챗봇", page_icon="📚", layout="wide"
    )
    load_dotenv(dotenv_path=ENV_PATH)
    url = os.getenv("SUPABASE_URL", "").strip()
    anon = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not anon:
        st.error("SUPABASE_URL 또는 SUPABASE_ANON_KEY가 설정되지 않았습니다.")
        st.stop()
    supabase = create_client(url, anon)
    render_header()
    schema_issue = app_users_schema_issue(supabase)
    if schema_issue:
        st.error("DB 스키마가 구버전이거나 마이그레이션이 중간 실패했습니다.")
        st.caption(f"원본 오류: {schema_issue}")
        st.markdown("Supabase SQL Editor에서 아래 SQL을 먼저 실행하세요.")
        st.code(
            "\n".join(
                [
                    "ALTER TABLE public.app_users ADD COLUMN IF NOT EXISTS password_hash text;",
                    "ALTER TABLE public.app_users ADD COLUMN IF NOT EXISTS password_salt text;",
                    "UPDATE public.app_users",
                    "SET password_hash = COALESCE(password_hash, ''),",
                    "    password_salt = COALESCE(password_salt, '')",
                    "WHERE password_hash IS NULL OR password_salt IS NULL;",
                    "ALTER TABLE public.app_users ALTER COLUMN password_hash SET NOT NULL;",
                    "ALTER TABLE public.app_users ALTER COLUMN password_salt SET NOT NULL;",
                ]
            ),
            language="sql",
        )
        st.info("또는 `7.MultiService/code/multi-users-ref.sql` 전체를 실행한 뒤 앱을 새로고침하세요.")
        st.stop()

    logged_in, app_user_id = auth_panel(supabase)
    if not logged_in or not app_user_id:
        st.info("사이드바에서 회원가입/로그인 후 사용하세요.")
        st.stop()

    with st.sidebar:
        st.markdown("### API Keys (사용자별 입력)")
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key")

        provider = st.selectbox("LLM Provider", ["openai", "anthropic", "gemini"], index=0)
        model_map = {
            "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
            "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
            "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
        }
        model_name = st.selectbox("LLM 모델", model_map[provider], index=0)

    if provider == "openai" and not openai_key.strip():
        st.warning("OpenAI 모델을 사용하려면 OpenAI API Key를 입력하세요.")
        st.stop()
    if provider == "anthropic" and not anthropic_key.strip():
        st.warning("Anthropic 모델을 사용하려면 Anthropic API Key를 입력하세요.")
        st.stop()
    if provider == "gemini" and not gemini_key.strip():
        st.warning("Gemini 모델을 사용하려면 Gemini API Key를 입력하세요.")
        st.stop()
    if not openai_key.strip():
        st.warning("RAG 임베딩/검색을 위해 OpenAI API Key는 항상 필요합니다.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_key.strip()
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key.strip()
    os.environ["GOOGLE_API_KEY"] = gemini_key.strip()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not st.session_state.get("current_session_id"):
        ensure_bootstrap(supabase, app_user_id)

    with st.sidebar:
        st.markdown("### 세션 관리")
        sessions = fetch_sessions(supabase, app_user_id)
        id_to_title = {str(s["id"]): s["title"] for s in sessions}

        def _label(sid: str) -> str:
            t = id_to_title.get(sid, sid)
            return f"{t} ({sid[:8]})"

        options = [str(s["id"]) for s in sessions]
        cur = st.session_state.current_session_id
        if cur not in options and options:
            cur = options[0]
            st.session_state.current_session_id = cur
            st.session_state.messages = fetch_messages(supabase, cur)

        picked = st.selectbox(
            "세션 선택 (선택 시 자동 로드)",
            options=options if options else [cur],
            index=(options.index(cur) if cur in options else 0),
            format_func=_label,
            key="session_select_widget",
        )
        if picked != st.session_state.current_session_id:
            st.session_state.current_session_id = picked
            st.session_state.messages = fetch_messages(supabase, picked)
            st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("세션저장"):
                if not st.session_state.messages:
                    st.warning("저장할 대화가 없습니다.")
                else:
                    llm_title = make_llm(provider, model_name, 0, False)
                    title = generate_session_title(llm_title, st.session_state.messages)
                    new_id = create_chat_session(supabase, app_user_id, title)
                    replace_session_messages(supabase, new_id, st.session_state.messages)
                    copy_vectors_between_sessions(supabase, st.session_state.current_session_id, new_id)
                    st.success(f"새 세션으로 저장했습니다: {title}")
                    st.rerun()

        with col_b:
            if st.button("세션로드"):
                sid = st.session_state.current_session_id
                st.session_state.messages = fetch_messages(supabase, sid)
                st.success("세션을 다시 불러왔습니다.")
                st.rerun()

        if st.button("세션삭제"):
            sid = st.session_state.current_session_id
            delete_chat_session(supabase, sid)
            sessions_after = fetch_sessions(supabase, app_user_id)
            if sessions_after:
                st.session_state.current_session_id = str(sessions_after[0]["id"])
                st.session_state.messages = fetch_messages(
                    supabase, st.session_state.current_session_id
                )
            else:
                nid = create_chat_session(supabase, app_user_id, "새 대화")
                st.session_state.current_session_id = nid
                st.session_state.messages = []
            st.success("세션을 삭제했습니다.")
            st.rerun()

        if st.button("화면초기화"):
            nid = create_chat_session(supabase, app_user_id, "새 대화")
            st.session_state.current_session_id = nid
            st.session_state.messages = []
            st.success("새 작업 화면으로 시작합니다.")
            st.rerun()

        if st.button("vectordb"):
            names = list_vector_filenames(supabase, st.session_state.current_session_id)
            st.text(
                "현재 세션 벡터 DB 파일명:\n" + "\n".join(names)
                if names
                else "현재 세션에 저장된 벡터 문서가 없습니다."
            )

        st.markdown("### PDF")
        files = st.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=True, key="pdf_up")
        if st.button("파일 처리하기") and files:
            try:
                process_uploaded_pdfs(
                    supabase, embeddings, st.session_state.current_session_id, list(files)
                )
                st.success("PDF 처리 및 벡터 저장이 완료되었습니다. (자동 저장)")
            except Exception as e:
                st.error(f"PDF 처리 오류: {e}")
                logging.exception("pdf")

        st.text(
            f"모델: {provider}:{model_name}\n"
            f"세션 ID: {st.session_state.current_session_id[:8]}…\n"
            f"메시지 수: {len(st.session_state.messages)}"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]), unsafe_allow_html=False)

    chat_q = st.chat_input("질문을 입력하세요.")
    if not chat_q:
        return

    llm = make_llm(provider, model_name, 0, True)
    llm_sync = make_llm(provider, model_name, 0, False)
    docs = retrieve_with_rpc(
        supabase, embeddings, st.session_state.current_session_id, app_user_id, chat_q, k=6
    )
    context = "\n\n".join(d.page_content for d in docs) if docs else "(관련 문서 없음)"

    lc_hist = to_lc_messages(st.session_state.messages)
    messages_for_llm = [
        SystemMessage(content=SYSTEM_PROMPT),
        *lc_hist,
        HumanMessage(content=f"참고 문맥:\n{context}\n\n사용자 질문:\n{chat_q}"),
    ]

    st.session_state.messages.append({"role": "user", "content": chat_q})
    with st.chat_message("user"):
        st.markdown(chat_q)

    with st.chat_message("assistant"):
        acc: list[str] = []

        def _stream() -> Iterator[str]:
            if provider == "gemini":
                # Gemini integration can be unstable with token streaming in 일부 환경.
                text = llm_invoke_text(llm, messages_for_llm)
                acc.append(text)
                yield text
                return
            for chunk in llm.stream(messages_for_llm):
                t = getattr(chunk, "content", "")
                if t:
                    acc.append(t)
                    yield t

        body = st.write_stream(_stream())
        main_text = body if isinstance(body, str) and body.strip() else "".join(acc)
        main_text = remove_separators(main_text)
        follow = generate_followup_block(llm_sync, chat_q, main_text, context)
        st.markdown(follow)
        final_answer = main_text + "\n\n" + follow

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    replace_session_messages(supabase, st.session_state.current_session_id, st.session_state.messages)

    if len([m for m in st.session_state.messages if m["role"] == "user"]) == 1:
        t = generate_session_title(llm_sync, st.session_state.messages)
        update_session_title(supabase, st.session_state.current_session_id, t)

    st.rerun()


if __name__ == "__main__":
    main()
