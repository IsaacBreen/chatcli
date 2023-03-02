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

T = TypeVar("T", Mapping, str)


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
    if isinstance(tree, Mapping):
        assert isinstance(delta, Mapping)
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


def chatgen(
    messages: Optional[List[Dict[str, str]]] = None,
    *,
    text_io: Optional[io.TextIOBase] = sys.stdout,
    sep: Optional[str] = "\n",
) -> Generator[Mapping, str, None]:
    """
    Chat with an OpenAI API model using a Python generator. Send your messages one at a time. The response is printed to the console.

    Parameters
    ----------
    messages : Optional[List[Dict[str, str]]], optional
        A list of messages to send to the model. Each message should be a dictionary with a "role" key and a "content" key. You should call `next` or `.send(None)` prior to sending your first message to prime the generator.
    text_io : Optional[io.TextIOBase], optional
        The text IO to write the model's response to, by default sys.stdout
    sep : Optional[str], optional
        The separator to write between messages, by default "\\n"

    Yields
    ------
    Mapping
        The model's response: a Mapping provided by the OpenAI API.
    """
    if messages is None:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Prime the generator
    user_message = yield

    while True:
        assert isinstance(user_message, str)

        # Add the user's message to the list of messages
        messages.append({"role": "user", "content": user_message})

        # Send the messages to the API, accumulate the responses, and print them as they come in
        response_accumulated = None

        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
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
            if text_io is not None and "content" in response.choices[0].delta:
                text_io.write(response.choices[0].delta.content)
                text_io.flush()

        # Apply the separator
        if text_io is not None:
            text_io.write(sep)
            text_io.flush()

        # Rename the delta to message
        response_accumulated.message = response_accumulated.pop("delta")

        # Get the next message from the user
        user_message = yield response_accumulated


def multiline_prompt(*, swap_newline_keys: bool) -> str:
    # Define the key bindings
    kb = KeyBindings()

    def enter(event: KeyPressEvent):
        event.current_buffer.insert_text("\n")

    def submit(event: KeyPressEvent):
        event.current_buffer.validate_and_handle()

    # Bind them
    if swap_newline_keys:
        kb.add("enter")(enter)
        kb.add("escape", "enter")(submit)
    else:
        kb.add("escape", "enter")(enter)
        kb.add("enter")(submit)

    # Define a prompt continuation function
    def prompt_continuation(width, line_number, wrap_count):
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
    chat = chatgen(messages)
    next(chat)

    # Get the first message from the user
    user_message = multiline_prompt(swap_newline_keys=swap_newline_keys)

    # Send the user's message to the generator
    while user_message != "exit":
        chat.send(user_message)
        user_message = multiline_prompt(swap_newline_keys=swap_newline_keys)


def test_chatgen() -> None:
    """
    Test the chatgen generator.
    """
    # Create the list of messages
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Create the generator
    chat = chatgen(messages)
    next(chat)

    # Send the user's message to the generator
    assert (
        "Dodgers"
        in chat.send("Who won the world series in 2020?")["message"]["content"]
    )
    assert "Texas" in chat.send("Where was it played?")["message"]["content"]


if __name__ == "__main__":
    fire.Fire(chatcli)
