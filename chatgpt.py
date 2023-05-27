import os
from datetime import datetime
from pathlib import Path
import pickle
import openai
import streamlit as st

API_KEY_ENV_NAME = "CHATGPT_APP_API_KEY"

SAVE_DIR = "savefiles"
BACKUP_NAME = "__backup__"

MODELS = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-3.5-turbo"]

MODEL_MAX_TOKENS = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-3.5-turbo": 4097}

DEFAULT_MODEL = "gpt-4"

REASON_ICONS = {
    "stop": "✅",
    "length": "💬",
    "content_filter": "✋",
    "null": "❓"}

DELETED_ICON = "🗑️"
SUMMARIZED_ICON = "📝"

INPUT_PLACEHOLDER = "{input}"

# TODO: English version
DEFAULT_PROMPT_TO_SUMMARIZE = """
ここまでの内容の要約を箇条書きで作成してください。
- 箇条書き以外の前置き、結論は不要です
- 時系列に沿って要約してください
- 項目数は最大でも10個程度にしてください
"""

MAX_SUMMARY_TOKENS = 512

os.makedirs(SAVE_DIR, exist_ok=True)

st.set_page_config(layout="wide")


SESSION_STATE = {
    "top_p": 0.4,
    "frequency_penalty": 1.0,
    "presence_penalty": 1.0,
    "messages": [],
    "last_usage": {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0},
    "basename": "",
    "template": INPUT_PLACEHOLDER,
    "summary": "",
    "summary_tokens": 0,
    "prompt_to_summerize": DEFAULT_PROMPT_TO_SUMMARIZE.strip(),
    "n_to_summerize": 7000 if "gpt-4" in DEFAULT_MODEL else 3000,
    "n_to_keep": 40 if "gpt-4" in DEFAULT_MODEL else 20,
    "n_before_summary": 1,
    "enable_auto_summarize": True,
    "last_request": {},
    "last_summary_request": {}}

# Alias
st_state = st.session_state

for k, v in SESSION_STATE.items():
    if k not in st_state:
        st_state[k] = v


def chatgpt(messages, api_key, model, top_p, frequency_penalty, presence_penalty, max_tokens):
    openai.api_key = api_key
    params = {
        "messages": messages,
        "model": model,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens}

    print("[API setting]")
    max_key_len = max(map(len, params.keys()))
    for k, v in params.items():
        if k == "messages":
            continue
        print(f"  {k.rjust(max_key_len)}: {v}")

    # This may throw exceptions such as timed out, exceeded token limits
    response = openai.ChatCompletion.create(**params)

    res_msg = response.choices[0].message
    return ({"role": res_msg["role"], "content": res_msg["content"].strip()},
            response.choices[0].finish_reason,
            response["usage"])


def insert_current_summary(messages, index):
    if st_state.summary.strip():
        current_summary = {"role": "system", "content": st_state.summary}
        messages.insert(index, current_summary)


def send_message(role, text, api_param):
    if role == "user" and INPUT_PLACEHOLDER in st_state.template:
        text_for_api = st_state.template.replace(INPUT_PLACEHOLDER, text)
        print("[User message after formatting]")
        print(text_for_api)
    else:
        text_for_api = text

    # Each message sent to ChatGPT should contain only the keys ChatGPT accepts ("role" and "content")
    msgs = [m for m in st_state.messages if not m["summarized"]]
    msgs = list(map(lambda x: {"role": x["role"], "content": x["content"]},
                    [*msgs, {"role": role, "content": text_for_api}]))

    insert_current_summary(msgs, st_state.n_before_summary)

    st_state.last_request = msgs

    # Kick API and receive a response from ChatGPT
    res, reason, usage = chatgpt(msgs,
                                 api_param["api_key"],
                                 api_param["model"],
                                 api_param["top_p"],
                                 api_param["frequency_penalty"],
                                 api_param["presence_penalty"],
                                 api_param["max_tokens"])

    # Update the history
    new_msg = {"role": role, "content": text.strip(), "delete_flag": False, "summarized": False}
    st_state.messages.append(new_msg)
    st_state.messages.append({**res, "finish_reason": reason, "delete_flag": False, "summarized": False})
    return usage


def summerize():
    if len(st_state.messages) <= st_state.n_to_keep:
        # Do nothing as all the messages should be kept
        return None, 0

    # Each message sent to ChatGPT should contain only the keys ChatGPT accepts ("role" and "content")
    msgs = [m for m in st_state.messages if not m["summarized"]]
    msgs = list(map(lambda x: {"role": x["role"], "content": x["content"]},
                    [*msgs[:-st_state.n_to_keep],
                    {"role": "system", "content": st_state.prompt_to_summerize}]))

    insert_current_summary(msgs, st_state.n_before_summary)

    st_state.last_summary_request = msgs

    # Kick API and receive a response from ChatGPT
    res, reason, usage = chatgpt(msgs,
                                 st_state.api_key,
                                 st_state.chatgpt_model,
                                 st_state.top_p,
                                 st_state.frequency_penalty,
                                 st_state.presence_penalty,
                                 MAX_SUMMARY_TOKENS)

    # TODO: Handle errors when the summary is too long
    if reason == "length":
        print("Summary is undone!")

    for i in range(len(st_state.messages) - st_state.n_to_keep):
        if i == 0:
            continue
        # Summarized messages won't be sent to ChatGPT
        st_state.messages[i]["summarized"] = True

    return res["content"], usage["completion_tokens"]


def save_messages(path_without_ext):
    # Serialize using pickle (to load later)
    with open(f"{path_without_ext}.pickle", 'wb') as p:
        pickle.dump({"messages": st_state.messages,
                     "summary": st_state.summary,
                     "template": st_state.template,
                     "model": st_state.chatgpt_model,
                     "top_p": st_state.top_p,
                     "freqency_penalty": st_state.frequency_penalty,
                     "presence_penalty": st_state.presence_penalty}, p)
    # Export as a markdown file
    msgs = [m for m in st_state.messages if not m["delete_flag"]]
    md = "\n\n---\n\n".join(map(lambda x: f"**{x['role']}**\n\n{x['content']}",
                                msgs))
    Path(f"{path_without_ext}.md").write_text(md + "\n")


with st.sidebar:
    st.title("API SETTING")
    with st.expander("API key"):
        api_key = os.getenv(API_KEY_ENV_NAME) or ""
        st.text_input("API key", api_key, key="api_key", label_visibility="collapsed")
    st.selectbox("Model", MODELS, index=MODELS.index(DEFAULT_MODEL), key="chatgpt_model")
    st.slider("top_p", 0.0, 1.0, step=0.1, key="top_p")
    st.slider("frequency_penalty", -2.0, 2.0, step=0.1, key="frequency_penalty")
    st.slider("presence_penalty", -2.0, 2.0, step=0.1, key="presence_penalty")

    st.selectbox("max_tokens",
                 (128, 256, 512, 1024, 2048),
                 index=2, key="max_tokens")

    st.title("SAVE/LOAD")
    # File basename (without extension) to save/load
    st.text_input("Basename", key="basename")

    def on_click_save():
        if st_state.basename.strip() == "":
            # When basename is empty, generate it from the current time
            gen_basename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S")
            st_state.basename = gen_basename

        save_messages(f"{SAVE_DIR}/{st_state.basename.strip()}")

    def on_click_load():
        # If basename is empty, use the auto backup file
        path = BACKUP_NAME + str(st_state.auto_save_slot) if st_state.basename.strip() == "" else f"{SAVE_DIR}/{st_state.basename}"

        with open(f"{path}.pickle", 'rb') as p:
            data = pickle.load(p)
            if type(data) is not dict:
                # Old format
                msgs = data
                summary = ""
                template = INPUT_PLACEHOLDER
                model = DEFAULT_MODEL
                top_p = 0.4
                frequency_penalty = 1.0
                presence_penalty = 1.0
            else:
                msgs = data["messages"]
                summary = data["summary"]
                template = data["template"]
                model = data.get("model", DEFAULT_MODEL)
                top_p = data.get("top_p", 0.4)
                frequency_penalty = data.get("frequency_penalty", 1.0)
                presence_penalty = data.get("presence_penalty", 1.0)

        # Old format
        for i in range(len(msgs)):
            if "delete_flag" not in msgs[i]:
                msgs[i]["delete_flag"] = False
            if "summarized" not in msgs[i]:
                msgs[i]["summarized"] = False

        st_state.messages = msgs
        st_state.last_usage = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0}
        st_state.summary = summary
        st_state.template = template
        st_state.chatgpt_model = model
        st_state.top_p = top_p
        st_state.frequency_penalty = frequency_penalty
        st_state.presence_penalty = presence_penalty

    col_save, col_load = st.columns(2)
    col_save.button("Save", on_click=on_click_save, use_container_width=True)
    col_load.button("Load", on_click=on_click_load, use_container_width=True)

    st.slider("Auto save slot", 0, 9, 0, 1, key="auto_save_slot")

    st.title("SUMMARY SETTING")
    st.checkbox("Enable auto summarize", key="enable_auto_summarize")

    st.number_input("N of tokens to summerize",
                    min_value=1024,
                    max_value=MODEL_MAX_TOKENS[st_state.chatgpt_model] - MAX_SUMMARY_TOKENS,
                    key="n_to_summerize")

    st.number_input("N of recent messages to keep",
                    min_value=2, max_value=60, key="n_to_keep")

    st.number_input("N of first messages before summary",
                    min_value=0, max_value=10, key="n_before_summary")


st.write("⚠️Always monitor your [API usage](https://platform.openai.com/account/usage) carefully. "
         "It's highly recommended to [setup usage limits](https://platform.openai.com/account/billing/limits)!")

# Template
with st.expander("Template"):
    # Template
    template_ph = st.empty()

# Prompt
input_text = st.text_area("Prompt", "")

# Actions
col_send_user, col_send_system = st.columns(2)

system_msg_ph = st.empty()

btn_send_user = col_send_user.button("Send as user", use_container_width=True)
btn_send_system = col_send_system.button("Send as system", use_container_width=True)

if btn_send_user:
    role = "user"
elif btn_send_system:
    role = "system"

try:
    if btn_send_user or btn_send_system:
        # Delete messages which delete_flag is True
        st_state.messages = list(filter(lambda x: not x["delete_flag"], st_state.messages))

        if st_state.enable_auto_summarize and st_state.last_usage["total_tokens"] >= st_state.n_to_summerize:
            with st.spinner("Summerizing..."):
                summary, summary_tokens = summerize()
                if summary is not None:
                    st_state.summary = summary
                    st_state.summary_tokens = summary_tokens
                print("Summary tokens:", summary_tokens)

        with st.spinner("Waiting for response from ChatGPT..."):
            api_param = {
                "api_key": st_state.api_key,
                "model": st_state.chatgpt_model,
                "top_p": st_state.top_p,
                "frequency_penalty": st_state.frequency_penalty,
                "presence_penalty": st_state.presence_penalty,
                "max_tokens": st_state.max_tokens}
            st_state.last_usage = send_message(role, input_text, api_param)
        save_messages(BACKUP_NAME + str(st_state.auto_save_slot))  # Automatically backup

except openai.InvalidRequestError as e:
    system_msg_ph.error(e)

template_ph.text_area("Template", key="template", label_visibility="collapsed")

tab_history, tab_summary, tab_last_req = st.tabs(["History", "Summary", "Last request"])

with tab_history:
    # Display token usage
    n_completion = st_state.last_usage["completion_tokens"]
    n_prompt = st_state.last_usage["prompt_tokens"]
    n_total = st_state.last_usage["total_tokens"]

    metric_cols = st.columns([1.5, 1.5, 1.5, 5.5])
    metric_cols[0].metric("Completion", n_completion)
    metric_cols[1].metric("Prompt", n_prompt)
    metric_cols[2].metric("Total (~4097)", n_total)

    # Message history
    fields = ["role", "content", "actions"]
    list_cols = st.columns([1, 6, 1])

    def deleted(txt):
        # Mark the text as deleted
        oneline = txt.replace("\n", "")
        return f"~~{oneline}~~"

    for col, field in zip(list_cols, fields):
        col.write(f"**{field}**")

    for i, msg in reversed(list(enumerate(st_state.messages))):
        role = msg["role"]
        if "finish_reason" in msg:
            icon = REASON_ICONS.get(msg["finish_reason"], "[?]")
            role = f"{role} {icon}"
        if msg["summarized"]:
            role = f"{role} {SUMMARIZED_ICON}"
        if msg["delete_flag"]:
            role = f"{role} {DELETED_ICON}"
        content = msg["content"]

        col_role, col_cont, col_act = st.columns([1, 6, 1])

        col_role_ph = col_role.empty()
        col_cont_pn = col_cont.empty()

        st.markdown("---")

        if msg["delete_flag"]:
            # Show the text as deleted
            col_role_ph.write(deleted(role))
            col_cont_pn.write(deleted(content))

            # Just show Edit button. It doesn't do anything
            col_act.button("Edit", key=f"edit_{i}")

        else:
            col_role_ph.write(role)  # Show the role as it is

            if col_act.button("Edit", key=f"edit_{i}"):
                # Switch to text area when Edit button is clicked
                if not msg["delete_flag"]:
                    key = f"txt_cont_{i}"

                    def get_updater(j, k):
                        # Make a handler as a closure to avoid because i, key will be overwritten
                        def update_messages():
                            st_state.messages[j]["content"] = st_state[k]
                        return update_messages

                    col_cont_pn.text_area("Content", content, key=key,
                                          on_change=get_updater(i, key),
                                          label_visibility="collapsed")

            else:
                # Show the content as it is
                col_cont_pn.write(content)

        if col_act.button("Del", key=f"del_{i}"):
            # Switch the delete flag
            msg["delete_flag"] = not msg["delete_flag"]
            col_role_ph.empty()
            col_cont_pn.empty()
            if msg["delete_flag"]:
                # Show the text as deleted
                role = f"{role} {DELETED_ICON}"
                col_role_ph.write(deleted(role))
                col_cont_pn.write(deleted(content))
            else:
                # Show the text as it is
                role = role.replace(f" {DELETED_ICON}", "")
                col_role_ph.write(role)
                col_cont_pn.write(content)

with tab_summary:
    with st.expander("Prompt to summerize"):
        # Template
        st.text_area("Prompt to summerize",
                     key="prompt_to_summerize", label_visibility="collapsed")

    st.text_area("Summary", "", key="summary")

with tab_last_req:
    with st.expander("Main chat"):
        st.json(st_state.last_request)
    with st.expander("Summary"):
        st.json(st_state.last_summary_request)
