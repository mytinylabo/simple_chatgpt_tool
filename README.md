# A simple tool for interacting with ChatGPT

## Getting started
1. Install dependencies: `pip install streamlit openai`.
2. Open `chatgpt.py` in your prefered editor and set your API key to `openai.api_key`.
3. Run the command bellow in your terminal and access the URL displayed:
```
streamlit run chatgpt.py
```

## Features
You can
* send your message as "user" or "system".
* adjust `top_p` and `max_tokens` settings for each message you send.
* save and load conversations. This generates two files: one is for loading later(`.pickle`) and another is for you to read(`.md`).
* edit messages in the conversation history, allowing you to direct the conversation as desired.

## Notes
* Icons displayed with "assistant" in the role column indicate [finish_reason](https://platform.openai.com/docs/guides/chat/response-format) of the corresponding messages:
  * ✅ stop
  * 💬 length
  * ✋ content_filter
  * ❓ null
* If the app crushes or freezes, simply reload and click "Load" button with "Basename" field left empty. The app will resume the last conversation.
