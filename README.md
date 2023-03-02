# ChatCLI

ChatCLI is a Python script that provides an easy-to-use Command Line Interface (CLI) for OpenAI's Chat API. ChatCLI aims to provide a similar experience the ChatGPT frontend, including streaming tokens from the model as they are generated.

```console
$ chatcli
ChatCLI v0.1.0 | ↩ : submit | meta + ↩ : newline
>>> Write 3 funny prompts for yourself.
1. If you could only communicate through interpretive dance for the next 24 hours, how would you go about your day?

2. You wake up in a world where everyone speaks in rhyme. How do you adapt to this unusual circumstance?

3. You can only speak in questions for the rest of the day. How do you navigate conversations with friends, co-workers, and strangers?
```

## Installation

```bash
pip install chatcli
```

## Usage

Run `chatcli` from the command line.

To see the available options, run `chatcli --help`.

```bash
$ chatcli --help
NAME
    chatcli.py - Chat with an OpenAI API model using the command line.

SYNOPSIS
    chatcli.py <flags>

DESCRIPTION
    Chat with an OpenAI API model using the command line.

FLAGS
    --system=SYSTEM
        Type: str
        Default: 'You are a helpful as...
        The system message to send to the model.
    -a, --assistant=ASSISTANT
        Type: Optional[Optional]
        Default: None
        The assistant message to send to the model.
    --swap_newline_keys=SWAP_NEWLINE_KEYS
        Type: bool
        Default: False
```

Once you start the script, you will be prompted to enter a message. Type your message and press the Enter key to send it to the OpenAI API model. The response from the model will be displayed on the screen.

## License

This software is licensed under the MIT License.
