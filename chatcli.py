"""
Streaming CLI interface for OpenAI's Chat API.
"""
import json
import logging
import os
import threading
import time
from enum import Enum
from functools import partial
from typing import *

import fire
import openai
import pkg_resources
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.auto_suggest import ThreadedAutoSuggest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer  # type: ignore

T = TypeVar("T")
U = TypeVar("U")
M = TypeVar("M", MutableMapping, str)

logging.debug('invisible magic')  # <-- magic (but annoying) fix for logging

logger = logging.getLogger(__name__)


class ChatGenerator:
    def __init__(
            self,
            messages: Optional[List[Dict[str, str]]] = None,
            *,
            sep: Optional[str] = "\n",
            model: Optional[str] = None,
    ) -> None:
        self.messages = messages if messages is not None else []
        self.sep = sep
        if model is None:
            # Get the model from the environment variable
            self.model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
            logger.info(f"Using model {self.model!r} from environment variable 'OPENAI_CHAT_MODEL'")
        else:
            self.model = model

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the list of messages.

        Args:
            role:
                The role of the message.
            content:
                The content of the message.
        """
        self.messages.append({"role": role, "content": content})

    def send(
            self, user_message: str, write: Callable[[str], T] = partial(print, end="")
    ) -> Dict[str, U]:
        """
        Send a user message to the API.

        Args:
            user_message:
                The user message to send to the API.
            write:
                A function to write a message to the console. It will be called each time a response is received from
                the API that modifies the content.

        Returns:
            The response from the API.
        """
        assert isinstance(user_message, str)

        # Add the user's message to the list of messages
        self.add_message("user", user_message)

        # Send the messages to the API, accumulate the responses, and print them as they come in
        response_accumulated = None

        def finalize():
            # Rename the delta to message
            response_accumulated.message = response_accumulated.pop("delta")

            # Apply the separator
            if self.sep is not None:
                write(self.sep)
                response_accumulated.message.content += self.sep

            # Add the response to the list of messages
            assert response_accumulated.message.role == "assistant"
            self.add_message("assistant", response_accumulated.message.content)

        try:
            logger.info(f"Sending messages to API: {json.dumps(self.messages, indent=2)}")
            for response in openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.messages,
                    stream=True,
            ):
                if response_accumulated is None:
                    # Set the initial response to the first response (less the delta)
                    response_accumulated = response.choices[0]
                else:
                    # Apply the delta to the accumulated response
                    response_accumulated.delta = string_tree_apply_delta(
                        response_accumulated.delta, response.choices[0].delta
                    )
                if "content" in response.choices[0].delta:
                    write(response.choices[0].delta.content)
        except KeyboardInterrupt:
            finalize()
            raise

        finalize()
        return response_accumulated

    def pop(self) -> MutableMapping:
        """
        Pop the last user message.
        """
        return self.messages.pop()


def string_tree_apply_delta(tree: M, delta: M) -> M:
    """
    Apply a delta to a tree of strings.

    Args:
        tree:
            The Python tree to apply the delta to.
        delta:
            The delta to apply to the Python tree.

    Returns:
        The Python tree with the delta applied.
    """
    if isinstance(tree, MutableMapping):
        assert isinstance(delta, MutableMapping)
        for key, value in delta.items():
            if key not in tree:
                tree[key] = value
            else:
                tree[key] = string_tree_apply_delta(tree[key], value)
        return tree
    elif isinstance(tree, str):
        assert isinstance(delta, str)
        return tree + delta
    else:
        raise TypeError(f"Invalid type {type(tree)}")


class PromptCode(Enum):
    UNDO = 1
    REDO = 2


def multiline_prompt(
        default: str = "", *, swap_newline_keys: bool, session: PromptSession, confirm_keyboard_interrupt: bool = True
) -> str:
    """
    Prompt the user for a multi-line input.

    Args:
        default:
            The default text to display in the prompt, by default ""
        swap_newline_keys:
            Whether to swap the keys for submitting and entering a newline.
        session:
            The prompt session to use.
        confirm_keyboard_interrupt:
            Whether to confirm a keyboard interrupt before raising it.

    Returns:
        The user's input.
    """
    # Define the key bindings
    kb = KeyBindings()

    def enter(event: KeyPressEvent):
        """
        Enter a newline.

        Args:
            event:
                The key press event.
        """
        event.current_buffer.insert_text("\n")

    def submit(event: KeyPressEvent):
        """
        Submit the input.

        Args:
            event:
                The key press event.
        """
        event.current_buffer.validate_and_handle()

    # Bind them
    if swap_newline_keys:
        kb.add("enter")(enter)
        kb.add("escape", "enter")(submit)
    else:
        kb.add("escape", "enter")(enter)
        kb.add("enter")(submit)

    # Add a key binding for undo
    @kb.add("c-z")
    def undo(event: KeyPressEvent):
        """
        Undo the last user message.

        Args:
            event:
                The key press event.
        """
        # Clear the prompt entered so far
        event.app.current_buffer.reset()
        # Exit the prompt with the UNDO code
        event.app.exit(result=PromptCode.UNDO)

    # Trigger suggestions on backspace
    @kb.add("backspace")
    def _(event: KeyPressEvent):
        """
        Trigger suggestions on backspace.

        Args:
            event:
                The key press event.
        """
        # Delete the last character
        if event.app.current_buffer.cursor_position != 0:
            event.app.current_buffer.cursor_position -= 1
            event.app.current_buffer.delete(1)
        event.app.current_buffer.complete_next()

    # Define a prompt continuation function
    def prompt_continuation(width: int, line_number: int, wrap_count: int) -> str:
        """
        Return the continuation prompt.

        Args:
            width:
                The width of the prompt.
            line_number:
                The line number of the prompt.
            wrap_count:
                The number of times the prompt has wrapped.

        Returns:
            The continuation prompt.
        """
        return "... ".rjust(width)

    while True:
        try:
            return session.prompt(
                ">>> ",
                default=default,
                multiline=True,
                key_bindings=kb,
                prompt_continuation=prompt_continuation,
            )
        except KeyboardInterrupt:
            if confirm_keyboard_interrupt:
                if not session.prompt(
                        "Are you sure you want to exit? [y/n] ",
                ) == "y":
                    session.output.cursor_up(2)
                    continue
            raise


class LLMAutoSuggest(AutoSuggest):
    def __init__(self, chat: ChatGenerator):
        self.chat = chat
        self.prev_completions = []
        self.num_prev_completions_to_keep = 10

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
        """
        Get the suggestion for the current buffer.

        Args:
            buffer:
                The current buffer.
            document:
                The current document.

        Returns:
            The suggestion for the current buffer.
        """
        for i, (prev_buffer_text, prev_completion) in enumerate(self.prev_completions):
            if prev_completion.startswith(buffer.text) and buffer.text.startswith(
                    prev_buffer_text
            ):
                # Move the previous completion to the front of the list
                self.prev_completions.insert(0, self.prev_completions.pop(i))
                suggestion = prev_completion[len(buffer.text):]
                # Return only the first line of the suggestion
                suggestion_lines = suggestion.splitlines()
                suggestion = suggestion_lines[0] if suggestion_lines else ""
                return Suggestion(suggestion)

        # Get the messages as a json
        chatcli_version = pkg_resources.get_distribution("chatcli").version
        messages_json = f"ChatCLI v{chatcli_version} | content format: JSON (escaped) | word-wrap: False\n"
        messages_json += "[\n  " + "".join(
            json.dumps(message) + ",\n  " for message in self.chat.messages
        )
        messages_json += '{"role": "user", "content": "' + json.dumps(buffer.text)[1:-1]
        stops = ['"}\n', '"},', "\n"]

        # Get the suggestion
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=messages_json,
            stop=stops,
            temperature=0.0,
            max_tokens=256,
        )
        try:
            suggestion = json.loads(f'"{response.choices[0].text}"')
        except json.JSONDecodeError:
            print(repr(response.choices[0].text))
            raise

        # Remove the stop token
        for stop in stops:
            if suggestion.endswith(stop):
                suggestion = suggestion[: -len(stop)]
                break

        # Save the completion
        if len(self.prev_completions) >= self.num_prev_completions_to_keep:
            self.prev_completions.pop()
        self.prev_completions.insert(0, (buffer.text, buffer.text + suggestion))

        # Return only the first line of the suggestion
        suggestion_lines = suggestion.splitlines()
        suggestion = suggestion_lines[0] if suggestion_lines else ""

        # Return the suggestion
        return Suggestion(suggestion)


def chatcli(
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        assistant: Optional[str] = None,
        swap_newline_keys: bool = False,
        autosuggest: Literal["llm", "history", "none"] = "history",
        verbose: bool = False,
) -> None:
    """
    Chat with an OpenAI API model using the command line.

    Args:
        messages:
            The messages to start the chat with.
        assistant:
            The assistant message to send to the model.
        swap_newline_keys:
            Whether to swap the keys for submitting and entering a newline.
        autosuggest:
            The autosuggester to use.
        verbose:
            Whether to print verbose output.
    """
    # If verbose is enabled, print logs
    if verbose:
        logger.setLevel(logging.INFO)
        logger.warn("Verbose mode may cause formatting issues due to the way chatcli erases and overwrites text using "
                    "ANSI escape sequences.")

    start_messages_default = [{"role": "system", "content": "You are a helpful assistant."}]

    # Print a header
    chatcli_version = pkg_resources.get_distribution("chatcli").version
    print(f"ChatCLI v{chatcli_version}")
    header = []
    if swap_newline_keys:
        header += ["[ALT/⌥] + [↩] to submit", "[↩] for newline"]
    else:
        header += ["[↩] to submit", "[ALT/⌥] + [↩] for newline"]
    header += ["[CTRL] + [C] to cancel generation or exit"]
    header += ["[CTRL] + [Z] to undo"]
    print(f"Instructions: {', '.join(header)}")

    # Create the list of messages
    if messages is None:
        messages = start_messages_default
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    # Create the generator
    chat = ChatGenerator(messages=messages)

    # Create the autosuggester
    if autosuggest == "llm":
        autosuggester = LLMAutoSuggest(chat=chat)
        auto_suggest = ThreadedAutoSuggest(autosuggester)
    elif autosuggest == "history":
        auto_suggest = AutoSuggestFromHistory()
    elif autosuggest == "none":
        auto_suggest = None
    else:
        raise ValueError(f"Invalid autosuggest: {autosuggest}")

    # Create prompt_toolkit objects
    session = PromptSession(
        lexer=PygmentsLexer(MarkdownLexer),
        auto_suggest=auto_suggest,
    )

    default = ""

    # This is the main loop
    while True:
        # Get the user's message
        user_message = multiline_prompt(
            swap_newline_keys=swap_newline_keys, session=session, default=default, confirm_keyboard_interrupt=True
        )

        next_default = ""

        def undo():
            nonlocal next_default
            # Undo the message
            while len(chat.messages) > 0 and chat.messages[-1]["role"] in [
                "user",
                "assistant",
            ]:
                message = chat.messages.pop()
                # Wind back the prompt
                for i in range(len(message["content"].splitlines())):
                    session.output.cursor_up(1)
                if message["role"] == "user":
                    next_default = message["content"]
                    break
            session.output.erase_down()
            session.output.flush()

        if user_message == PromptCode.UNDO:
            # Undo the prompt
            session.output.cursor_up(1)
            undo()

        elif user_message == "exit":
            break
        else:
            assert isinstance(user_message, str)

            def write(message: str) -> None:
                """
                Write a message to the console.
                """
                session.output.write(message)
                session.output.flush()

            class AsyncSmoothWriter:
                def __init__(self, write, flush_interval=0.1, poll_interval=0.01):
                    self._write = write
                    self.flush_interval = flush_interval
                    self.poll_interval = poll_interval
                    self.buffer = ""
                    self.lock = threading.Lock()
                    self.should_stop = False
                    self.thread = threading.Thread(target=self._run)
                    self.thread.start()

                def _run(self):
                    while True:
                        with self.lock:
                            if len(self.buffer) == 0:
                                if self.should_stop:
                                    break
                                wait_time = self.poll_interval
                            else:
                                char = self.buffer[0]
                                wait_time = self.flush_interval / len(self.buffer)
                                self.buffer = self.buffer[1:]
                                self._write(char)
                        time.sleep(wait_time)

                def write(self, message):
                    with self.lock:
                        self.buffer += message

                def wait_until_buffer_empty(self):
                    with self.lock:
                        self.should_stop = True
                    self.thread.join()

            # Create the smooth writer
            smooth_writer = AsyncSmoothWriter(write)

            # Send the user's message to the generator
            try:
                chat.send(user_message, write=smooth_writer.write)
            except KeyboardInterrupt:
                # Wait for the buffer to empty
                smooth_writer.wait_until_buffer_empty()
                undo()
            # Wait for the buffer to empty
            smooth_writer.wait_until_buffer_empty()

        # Set the default
        default = next_default


def test_chatgen() -> None:
    """
    Test the chatgen generator.
    """
    # Create the list of messages
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Create the generator
    chat = ChatGenerator(messages)

    # Send the user's message to the generator
    assert (
            "Dodgers"
            in chat.send("Who won the world series in 2020?", write=print)["message"][
                "content"
            ]
    )
    assert (
            "Texas" in chat.send("Where was it played?", write=print)["message"]["content"]
    )


if __name__ == "__main__":
    fire.Fire(chatcli)
