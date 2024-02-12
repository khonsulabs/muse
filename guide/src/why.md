# Why Muse over ...?

Muse's main distinguishing factor is that it attempts to provide a safe way to
execute untrusted code, including code that may never terminate.

All languages have different design goals, performance, and developer
experiences. In the end, there are many viable options for embedded languages.
The choice of what language to embed is a highly personal decision.

## Why use Muse over WASM?

Overall, WASM is a very tempting option for safely executing untrusted code
within another application. Embedding WASM allows users to run arbitrary code in
isolated environments incredibly efficiently and safely. Many runtimes also
support budgeted execution to allow developers to defend against inefficient
code or infinite loops. It used to be a fair amount of work to set up a plugin
system using WASM, but [Extism](https://extism.org/) makes it incredibly easy
today.

The downsides *might* be:

* Large number of dependencies. Let's face it, build times can sometimes be an
  issue. Muse has limited dependencies, and due to being a very focused
  implementation, it means Muse will always be less code than embedding a WASM
  runtime.
* Serialization Overhead. With Muse, your Rust types can be used directly within
  the scripts with no serialization overhead. With WASM, because memory isn't
  guaranteed to be compatible, serialization is often used when passing data
  between native code and WASM.
* Split documentation. With every language that your plugin system supports, you
  might feel the need to provide documentation for those languages. By focusing
  on a single embedded language, the end-user documentation may be less work to
  maintain.

Ultimately if the allure of having users be able to write code in the language
they are most familiar with, WASM is a great option.
