"""
Streaming CLI interface for OpenAI's Chat API.
"""
import json
from enum import Enum
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
M = TypeVar("M", MutableMapping, str)


class ChatGenerator:
    def __init__(
            self,
            messages: Optional[List[Dict[str, str]]] = None,
            *,
            sep: Optional[str] = "\n",
    ) -> None:
        self.messages = messages or [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.sep = sep

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

    def send(self, user_message: str, write: Callable[[str], None]) -> Dict[str, T]:
        """
        Send a user message to the API.

        Args:
            user_message:
                The user message to send to the API.
            write:
                A function to write a message to the console.

        Returns:
            The response from the API.
        """
        assert isinstance(user_message, str)

        # Add the user's message to the list of messages
        self.add_message("user", user_message)

        # Send the messages to the API, accumulate the responses, and print them as they come in
        response_accumulated = None

        for response in openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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

        # Rename the delta to message
        response_accumulated.message = response_accumulated.pop("delta")

        # Apply the separator
        if self.sep is not None:
            write(self.sep)
            response_accumulated.message.content += self.sep

        # Add the response to the list of messages
        assert response_accumulated.message.role == "assistant"
        self.add_message("assistant", response_accumulated.message.content)

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
        default: str = "", *, swap_newline_keys: bool, session: PromptSession
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
    # @kb.add("backspace")
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

    return session.prompt(
        ">>> ",
        default=default,
        multiline=True,
        key_bindings=kb,
        prompt_continuation=prompt_continuation,
    )


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
        system: str = "You are a helpful assistant.",
        assistant: Optional[str] = None,
        swap_newline_keys: bool = False,
        autosuggest: Literal["llm", "history", "none"] = "llm",
) -> None:
    """
    Chat with an OpenAI API model using the command line.

    Args:
        system:
            The system message to send to the model.
        assistant:
            The assistant message to send to the model.
        swap_newline_keys:
            Whether to swap the keys for submitting and entering a newline.
    """

    # Print a header
    chatcli_version = pkg_resources.get_distribution("chatcli").version
    print(f"ChatCLI v{chatcli_version}", end=" | ")
    if swap_newline_keys:
        print("meta + ↩ submit | ↩ newline")
    else:
        print("↩ : submit | meta + ↩ : newline")

    # Create the list of messages
    messages = [{"role": "system", "content": system}]
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
            swap_newline_keys=swap_newline_keys, session=session, default=default
        )

        next_default = ""

        if user_message == PromptCode.UNDO:
            # Undo the prompt
            session.output.cursor_up(1)
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

        elif user_message == "exit":
            break
        else:
            assert isinstance(user_message, str)

            def write(message: str) -> None:
                """
                Write a message to the console.
                """
                session.output.write_raw(message)
                session.output.flush()

            # Send the user's message to the generator
            chat.send(user_message, write=write)

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