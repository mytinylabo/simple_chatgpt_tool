import os
from datetime import datetime
from pathlib import Path
import pickle
import openai
import streamlit as st

# Set your API key here. NEVER share your key with others!!
openai.api_key = "..."

DEFAULT_TOP_P = 0.4
DEFAULT_MAX_TOKENS = 2  # Index of (128, 256, 512, 1024, 2048)

SAVE_DIR = "savefiles"
BACKUP_NAME = "__backup__"

MODEL = "gpt-3.5-turbo"

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
- 項目数は最大でも10個程度にしてください
"""

MAX_SUMMARY_TOKENS = 512

os.makedirs(SAVE_DIR, exist_ok=True)

st.set_page_config(layout="wide")


def chatgpt(messages, top_p, max_tokens):
    params = {
        "messages": messages,
        "model": MODEL,
        "top_p": top_p,
        "max_tokens": max_tokens}

    # TODO: Handle errors such as timed out, exceeded token limits
    response = openai.ChatCompletion.create(**params)

    res_msg = response.choices[0].message
    return ({"role": res_msg["role"], "content": res_msg["content"].strip()},
            response.choices[0].finish_reason,
            response["usage"])


def insert_current_summary(messages, index):
    if st.session_state.summary.strip():
        current_summary = {"role": "system", "content": st.session_state.summary}
        messages.insert(index, current_summary)


def send_message(role, text, top_p, max_tokens):
    if role == "user" and INPUT_PLACEHOLDER in st.session_state.template:
        text_for_api = st.session_state.template.replace(INPUT_PLACEHOLDER, text)
        print("User message after formatting:")
        print(text_for_api)
    else:
        text_for_api = text

    # Each message sent to ChatGPT should contain only the keys ChatGPT accepts ("role" and "content")
    msgs = [m for m in st.session_state.messages if not m["summarized"]]
    msgs = list(map(lambda x: {"role": x["role"], "content": x["content"]},
                    [*msgs, {"role": role, "content": text_for_api}]))

    insert_current_summary(msgs, st.session_state.n_before_summary)

    # Kick API and receive a response from ChatGPT
    res, reason, usage = chatgpt(msgs, top_p, max_tokens)

    # Update the history
    new_msg = {"role": role, "content": text.strip(), "delete_flag": False, "summarized": False}
    st.session_state.messages.append(new_msg)
    st.session_state.messages.append({**res, "finish_reason": reason, "delete_flag": False, "summarized": False})
    return usage


def summerize():
    if len(st.session_state.messages) <= st.session_state.n_to_keep:
        # Do nothing as all the messages should be kept
        return None, 0

    # Each message sent to ChatGPT should contain only the keys ChatGPT accepts ("role" and "content")
    msgs = [m for m in st.session_state.messages if not m["summarized"]]
    msgs = list(map(lambda x: {"role": x["role"], "content": x["content"]},
                    [*st.session_state.messages[:-st.session_state.n_to_keep],
                    {"role": "system", "content": st.session_state.prompt_to_summerize}]))

    insert_current_summary(msgs, st.session_state.n_before_summary)

    # Kick API and receive a response from ChatGPT
    res, reason, usage = chatgpt(msgs, top_p, MAX_SUMMARY_TOKENS)

    # TODO: Handle errors when the summary is too long
    if reason == "length":
        print("Summary is undone!")

    for i in range(len(st.session_state.messages) - st.session_state.n_to_keep):
        if i == 0:
            continue
        # Summarized messages won't be sent to ChatGPT
        st.session_state.messages[i]["summarized"] = True

    return res["content"], usage["completion_tokens"]


if "messages" not in st.session_state:
    # Message history
    st.session_state.messages = []
if "last_usage" not in st.session_state:
    # Token usage of the last interaction with ChatGPT
    st.session_state.last_usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0}
if "basename" not in st.session_state:
    # Basename used to save messages
    st.session_state.basename = ""
if "template" not in st.session_state:
    st.session_state.template = INPUT_PLACEHOLDER
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "summary_tokens" not in st.session_state:
    st.session_state.summary_tokens = 0
if "prompt_to_summerize" not in st.session_state:
    st.session_state.prompt_to_summerize = DEFAULT_PROMPT_TO_SUMMARIZE.strip()
if "n_to_summerize" not in st.session_state:
    st.session_state.n_to_summerize = 3000
if "n_to_keep" not in st.session_state:
    st.session_state.n_to_keep = 20
if "n_before_summary" not in st.session_state:
    st.session_state.n_before_summary = 1
if "enable_auto_summarize" not in st.session_state:
    st.session_state.enable_auto_summarize = True

st.write("⚠️Always monitor your [API usage](https://platform.openai.com/account/usage) carefully. "
         "It's highly recommended to [setup usage limits](https://platform.openai.com/account/billing/limits)!")

# Template
with st.expander("Template"):
    # Template
    template_ph = st.empty()

# Prompt
input_text = st.text_area("Prompt", "")

# Actions
action_cols = st.columns([1, 1, 2, 1, 3, 1, 1])

top_p = action_cols[2].slider("top_p", 0.0, 1.0, DEFAULT_TOP_P, step=0.1)

max_tokens = action_cols[3].selectbox(
    "max_tokens",
    (128, 256, 512, 1024, 2048), index=DEFAULT_MAX_TOKENS)

system_msg_ph = st.empty()


def save_messages(path_without_ext):
    # Serialize using pickle (to load later)
    with open(f"{path_without_ext}.pickle", 'wb') as p:
        pickle.dump({"messages": st.session_state.messages,
                     "summary": st.session_state.summary,
                     "template": st.session_state.template}, p)
    # Export as a markdown file
    msgs = [m for m in st.session_state.messages if not m["delete_flag"]]
    md = "\n\n---\n\n".join(map(lambda x: f"**{x['role']}**\n\n{x['content']}",
                                msgs))
    Path(f"{path_without_ext}.md").write_text(md + "\n")


btn_send_user = action_cols[0].button("Send as user")
btn_send_system = action_cols[1].button("Send as system")

if btn_send_user:
    role = "user"
elif btn_send_system:
    role = "system"

try:
    if btn_send_user or btn_send_system:
        # Delete messages which delete_flag is True
        st.session_state.messages = list(filter(lambda x: not x["delete_flag"], st.session_state.messages))

        if st.session_state.enable_auto_summarize and st.session_state.last_usage["total_tokens"] >= st.session_state.n_to_summerize:
            with st.spinner("Summerizing..."):
                summary, summary_tokens = summerize()
                if summary is not None:
                    st.session_state.summary = summary
                    st.session_state.summary_tokens = summary_tokens
                print("Summary tokens:", summary_tokens)

        with st.spinner("Waiting for response from ChatGPT..."):
            st.session_state.last_usage = send_message(role, input_text, top_p, max_tokens)
        save_messages(BACKUP_NAME)  # Automatically backup

except openai.InvalidRequestError as e:
    system_msg_ph.error(e)

# File basename (without extension) to save/load
input_basename = action_cols[4].text_input("Basename", key="basename")


def on_click_save():
    if st.session_state.basename.strip() == "":
        # When basename is empty, generate it from the current time
        gen_basename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S")
        st.session_state.basename = gen_basename

    save_messages(f"{SAVE_DIR}/{st.session_state.basename}")


# Save button
# Use a callback to work with session_state
action_cols[5].button("Save", on_click=on_click_save)

if action_cols[6].button("Load"):
    # If basename is empty, use the auto backup file
    path = BACKUP_NAME if input_basename.strip() == "" else f"{SAVE_DIR}/{input_basename}"

    with open(f"{path}.pickle", 'rb') as p:
        data = pickle.load(p)
        if type(data) is not dict:
            # Old format
            msgs = data
            summary = ""
            template = INPUT_PLACEHOLDER
        else:
            msgs = data["messages"]
            summary = data["summary"]
            template = data["template"]

    # Old format
    for i in range(len(msgs)):
        if "delete_flag" not in msgs[i]:
            msgs[i]["delete_flag"] = False
        if "summarized" not in msgs[i]:
            msgs[i]["summarized"] = False

    st.session_state.messages = msgs
    st.session_state.last_usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0}
    st.session_state.summary = summary
    st.session_state.template = template

template_ph.text_area("Template", "", key="template", label_visibility="collapsed")

tab_history, tab_summary = st.tabs(["History", "Summary"])

with tab_history:
    # Display token usage
    n_completion = st.session_state.last_usage["completion_tokens"]
    n_prompt = st.session_state.last_usage["prompt_tokens"]
    n_total = st.session_state.last_usage["total_tokens"]

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

    for i, msg in reversed(list(enumerate(st.session_state.messages))):
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
                            st.session_state.messages[j]["content"] = st.session_state[k]
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
                col_role_ph.write(deleted(role))
                col_cont_pn.write(deleted(content))
            else:
                # Show the text as it is
                col_role_ph.write(role)
                col_cont_pn.write(content)

with tab_summary:
    with st.expander("Prompt to summerize"):
        # Template
        st.text_area("Prompt to summerize", "",
                     key="prompt_to_summerize", label_visibility="collapsed")

    summary_cols = st.columns([1, 1.5, 1.5, 1.5, 4.5])

    summary_cols[0].checkbox("Enable auto summarize", key="enable_auto_summarize")

    summary_cols[1].number_input("N of tokens to summerize",
                                 min_value=1024, max_value=4097 - MAX_SUMMARY_TOKENS, key="n_to_summerize")

    summary_cols[2].number_input("N of recent messages to keep",
                                 min_value=2, max_value=30, key="n_to_keep")

    summary_cols[3].number_input("N of first messages before summary",
                                 min_value=0, max_value=10, key="n_before_summary")

    st.text_area("Summary", "", key="summary")
