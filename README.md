# ChatCLI

ChatCLI is a command line interface for OpenAI's Chat API.

 ChatCLI aims to provide a similar experience the ChatGPT frontend; like ChatGPT, ChatCLI streams tokens from the model as they are generated.

```console
$ chatcli
ChatCLI v0.1.5 | [↩] to submit | [ALT/⌥] + [↩] for newline
>>> Write 3 funny prompts for yourself.
1. If you could only communicate through interpretive dance for the next 24 hours, how would you go about your day?

2. You wake up in a world where everyone speaks in rhyme. How do you adapt to this unusual circumstance?

3. You can only speak in questions for the rest of the day. How do you navigate conversations with friends, co-workers, and strangers?
```

## Installation

To install ChatCLI, use pip:

```
pip install chatcli
```

You'll need an OpenAI API key to use ChatCLI. You can get one [here](https://beta.openai.com/).

ChatCLI reads your key from the `OPENAI_API_KEY` environment variable, so you'll also need to set that.

## Usage

### CLI Usage

To use ChatCLI, simply run the `chatcli` command:

```
chatcli
```

This will start a chat session with the default OpenAI model. You can type your messages and the AI will respond.

You can exit the chat at any time by typing "exit" into the prompt or pressing Ctrl+C.

### Python Usage

ChatCLI can also be used as a Python library. Here's an example:

```python
import chatcli

# Create the chat generator
generator = chatcli.ChatGenerator()

# Send a message to the generator
response = generator.send("Prove Riemann's Hypothesis.")

# Print the response
print(response["message"]["content"])
```

This will send a message to the API and then print the response stream as it is generated.

## License

ChatCLI is licensed under the MIT license. See [LICENSE](LICENSE) for details.
