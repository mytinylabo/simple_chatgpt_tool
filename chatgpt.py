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

INPUT_PLACEHOLDER = "{input}"

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


def send_message(role, text, top_p, max_tokens):
    # Delete messages which delete_flag is True
    st.session_state.messages = list(filter(lambda x: not x["delete_flag"], st.session_state.messages))

    if role == "user" and INPUT_PLACEHOLDER in st.session_state.template:
        text_for_api = st.session_state.template.replace(INPUT_PLACEHOLDER, text)
        print("User message after formatting:")
        print(text_for_api)
    else:
        text_for_api = text

    # Each message sent to ChatGPT should contain only the keys ChatGPT accepts ("role" and "content")
    msgs = list(map(lambda x: {"role": x["role"], "content": x["content"]},
                    [*st.session_state.messages, {"role": role, "content": text_for_api}]))

    # Kick API and receive a response from ChatGPT
    res, reason, usage = chatgpt(msgs, top_p, max_tokens)

    # Update the history
    new_msg = {"role": role, "content": text.strip(), "delete_flag": False}
    st.session_state.messages.append(new_msg)
    st.session_state.messages.append({**res, "finish_reason": reason, "delete_flag": False})
    return usage


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

st.write("⚠️Always monitor your [API usage](https://platform.openai.com/account/usage) carefully. "
         "It's highly recommended to [setup usage limits](https://platform.openai.com/account/billing/limits)!")

# Template
with st.expander("Template"):
    # Template
    input_template = st.text_area("Template", "", key="template", label_visibility="collapsed")

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
        pickle.dump(st.session_state.messages, p)
    # Export as a markdown file
    md = "\n\n---\n\n".join(map(lambda x: f"**{x['role']}**\n\n{x['content']}",
                                st.session_state.messages))
    Path(f"{path_without_ext}.md").write_text(md + "\n")


try:
    # Send the prompt as user
    if action_cols[0].button("Send as user"):
        with st.spinner("Waiting for response from ChatGPT..."):
            st.session_state.last_usage = send_message("user", input_text, top_p, max_tokens)
        save_messages(BACKUP_NAME)  # Automatically backup

    # Send the prompt as system
    if action_cols[1].button("Send as system"):
        with st.spinner("Waiting for response from ChatGPT..."):
            st.session_state.last_usage = send_message("system", input_text, top_p, max_tokens)
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
        msgs = pickle.load(p)
    st.session_state.messages = msgs
    st.session_state.last_usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0}

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
    return f"~~{oneline}~~ 🗑️"


for col, field in zip(list_cols, fields):
    col.write(f"**{field}**")

for i, msg in reversed(list(enumerate(st.session_state.messages))):
    role = msg["role"]
    if "finish_reason" in msg:
        icon = REASON_ICONS.get(msg["finish_reason"], "[?]")
        role = f"{role} {icon}"
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
