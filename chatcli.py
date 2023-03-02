"""
Streaming CLI interface for OpenAI's Chat API.
"""
import io
import sys
from typing import *
import pkg_resources
import fire
import openai
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding import KeyPressEvent
from prompt_toolkit.shortcuts import prompt

T = TypeVar("T", MutableMapping, str)


class ChatGenerator:
    def __init__(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        *,
        text_io: Optional[io.TextIOBase] = sys.stdout,
        sep: Optional[str] = "\n",
    ) -> None:
        self.messages = messages or [{"role": "system", "content": "You are a helpful assistant."}]
        self.text_io = text_io or sys.stdout
        self.sep = sep

    def __call__(self, user_message: str) -> MutableMapping:
        assert isinstance(user_message, str)

        # Add the user's message to the list of messages
        self.messages.append({"role": "user", "content": user_message})

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
            if self.text_io is not None and "content" in response.choices[0].delta:
                self.text_io.write(response.choices[0].delta.content)
                self.text_io.flush()

        # Apply the separator
        if self.text_io is not None:
            self.text_io.write(self.sep)
            self.text_io.flush()

        # Rename the delta to message
        response_accumulated.message = response_accumulated.pop("delta")

        # Get the next message from the user
        return response_accumulated


def string_tree_apply_delta(tree: T, delta: T) -> T:
    """
    Apply a delta to a tree of strings.

    Parameters
    ----------
    tree : T
        The Python tree to apply the delta to.
    delta : T
        The delta to apply to the Python tree.

    Returns
    -------
    T
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


def multiline_prompt(*, swap_newline_keys: bool) -> str:
    """
    Prompt the user for a multi-line input.

    Parameters
    ----------
    swap_newline_keys : bool
        Whether to swap the keys for submitting and entering a newline.

    Returns
    -------
    str
        The user's input.
    """
    # Define the key bindings
    kb = KeyBindings()

    def enter(event: KeyPressEvent):
        """
        Enter a newline.

        Parameters
        ----------
        event : KeyPressEvent
            The key press event.
        """
        event.current_buffer.insert_text("\n")

    def submit(event: KeyPressEvent):
        """
        Submit the input.

        Parameters
        ----------
        event : KeyPressEvent
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

    # Define a prompt continuation function
    def prompt_continuation(width: int, line_number: int, wrap_count: int) -> str:
        """
        Return the continuation prompt.

        Parameters
        ----------
        width : int
            The width of the prompt.
        line_number : int
            The line number of the prompt.
        wrap_count : int
            The number of times the prompt has wrapped.

        Returns
        -------
        str
            The continuation prompt.
        """
        return "... ".rjust(width)

    return prompt(
        ">>> ", multiline=True, key_bindings=kb, prompt_continuation=prompt_continuation
    )


def chatcli(
    *,
    system: str = "You are a helpful assistant.",
    assistant: Optional[str] = None,
    swap_newline_keys: bool = False,
) -> None:
    """
    Chat with an OpenAI API model using the command line.

    Parameters
    ----------
    system : str
        The system message to send to the model.
    assistant : Optional[str]
        The assistant message to send to the model.
    swap_newline_keys : bool
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
    chat = ChatGenerator(messages)

    # Get the first message from the user
    user_message = multiline_prompt(swap_newline_keys=swap_newline_keys)

    # Send the user's message to the generator
    while user_message != "exit":
        chat(user_message)
        user_message = multiline_prompt(swap_newline_keys=swap_newline_keys)


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
        in chat("Who won the world series in 2020?")["message"]["content"]
    )
    assert "Texas" in chat("Where was it played?")["message"]["content"]


if __name__ == "__main__":
    fire.Fire(chatcli)
