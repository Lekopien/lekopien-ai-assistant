"""Chat with the assistant in your terminal.  python cli.py [--llm] [--save chat.json]"""
import argparse

from assistant import Assistant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="use a local Ollama model if it is running")
    parser.add_argument("--save", help="write the chat history to this JSON file on exit")
    args = parser.parse_args()

    bot = Assistant(use_llm=args.llm)
    print("Lekopien assistant. Type 'quit' to exit.")
    try:
        while (msg := input("you> ").strip()).lower() not in {"quit", "exit"}:
            if msg:
                r = bot.ask(msg)
                print(f"bot> {r.text}\n     [{r.intent} {r.confidence:.0%} via {r.source}]")
    except (EOFError, KeyboardInterrupt):
        print()
    if args.save:
        bot.save_history(args.save)
        print(f"saved {len(bot.history)} turns to {args.save}")


if __name__ == "__main__":
    main()
