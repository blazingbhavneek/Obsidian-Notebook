A **function** is a reusable sequence of statements designed to do a particular job.

The function initiating the function call is the **caller**, and the function being **called** (executed) is the **callee**. A function call is also sometimes called an **invocation**, with the caller **invoking** the callee.

The first line is informally called the **function header**, and it tells the compiler about the existence of a function, the function’s name, and some other information 


“foo” is a meaningless word that is often used as a placeholder name for a function or variable when the name is unimportant to the demonstration of some concept. Such words are called [metasyntactic variables](https://en.wikipedia.org/wiki/Metasyntactic_variable) (though in common language they’re often called “placeholder names”

```cpp
#include <iostream>

int getValueFromUser() // this function now returns an integer value
{
 	std::cout << "Enter an integer: ";
	int input{};
	std::cin >> input;

	return input; // return the value the user entered back to the caller
}

int main()
{
	int num { getValueFromUser() }; // initialize num with the return value of getValueFromUser()

	std::cout << num << " doubled is: " << num * 2 << '\n';

	return 0;
}
```

> See the way the variable is init with curly braces

C does allow `main()` to be called explicitly, so some C++ compilers will allow this for compatibility reasons.

It is a common misconception that `main` is always the first function that executes.

Global variables are initialized prior to the execution of `main`. If the initializer for such a variable invokes a function, then that function will execute prior to `main`. We discuss global variables in lesson [7.4 -- Introduction to global variables](https://www.learncpp.com/cpp-tutorial/introduction-to-global-variables/).

You may be wondering why we return 0 from `main()`, and when we might return something else.

The return value from `main()` is sometimes called a **status code** (or less commonly, an **exit code**, or rarely a **return code**). The status code is used to signal whether your program was successful or not. By convention, a status code of `0` means the program ran normally (meaning the program executed and behaved as expected).

The C++ standard only defines the meaning of 3 status codes: `0`, `EXIT_SUCCESS`, and `EXIT_FAILURE`. `0` and `EXIT_SUCCESS` both mean the program executed successfully. `EXIT_FAILURE` means the program did not execute successfully.

`EXIT_SUCCESS` and `EXIT_FAILURE` are preprocessor macros defined in the \<cstdlib\> header:

```cpp
#include <cstdlib> // for EXIT_SUCCESS and EXIT_FAILURE

int main()
{
    return EXIT_SUCCESS;
}
```

If you want to maximize portability, you should only use `0` or `EXIT_SUCCESS` to indicate a successful termination, or `EXIT_FAILURE` to indicate an unsuccessful termination.


A value-returning function that does not return a value will produce undefined behavior
Function main will implicitly return 0 if no return statement is provided

In most cases, compilers will detect if you’ve forgotten to return a value. However, in some complicated cases, the compiler may not be able to properly determine whether your function returns a value or not in all cases


A value-returning function can only return a single value back to the caller each time it is called.

Note that the value provided in a return statement doesn’t need to be literal -- it can be the result of any valid expression, including a variable or even a call to another function that returns a value. In the `getValueFromUser()` example above, we returned a variable `input`, which held the number the user input.

There are various ways to work around the limitation of functions only being able to return a single value, which we’ll cover in future lessons.


Best practice

Do not put a return statement at the end of a non-value returning function.


Some statements require values to be provided, and others don’t.

When we have a statement that consists of just a function call (e.g. the first `printHi()` in the above example), we’re calling a function for its behavior, not its return value. In this case, we can call either a non-value returning function, or we can call a value-returning function and just ignore the return value.


Unreferenced parameters and unnamed parameters
Just like with unused local variables, your compiler will probably warn that variable `count` has been defined but not used.

In a function definition, the name of a function parameter is optional. Therefore, in cases where a function parameter needs to exist but is not used in the body of the function, you can simply omit the name. A parameter without a name is called an **unnamed parameter**:


You’re probably wondering why we’d write a function that has a parameter whose value isn’t used. This happens most often in cases similar to the following:

1. Let’s say we have a function with a single parameter. Later, the function is updated in some way, and the value of the parameter is no longer needed. If the now-unused function parameter were simply removed, then every existing call to the function would break (because the function call would be supplying more arguments than the function could accept). This would require us to find every call to the function and remove the unneeded argument. This might be a lot of work (and require a lot of retesting). It also might not even be possible (in cases where we did not control all of the code calling the function). So instead, we might leave the parameter as it is, and just have it do nothing.


Explain this? Operators `++` and `--` have prefix and postfix variants (e.g. `++foo` vs `foo++`). An unreferenced function parameter is used to differentiate whether an overload of such an operator is for the prefix or postfix case. We cover this in lesson [21.8 -- Overloading the increment and decrement operators](https://www.learncpp.com/cpp-tutorial/overloading-the-increment-and-decrement-operators/). - When we need to determine something from the type (rather than the value) of a type template parameter.

Best practice

When a function parameter exists but is not used in the body of the function, do not give it a name. You can optionally put a name inside a comment.

Local variable lifetime

**lifetime** is defined to be the time between its creation and destruction. Note that variable creation and destruction happen when the program is running (called runtime), not at compile time. Therefore, lifetime is a runtime property.

The above rules around creation, initialization, and destruction are guarantees. That is, objects must be created and initialized no later than the point of definition, and destroyed no earlier than the end of the set of the curly braces in which they are defined (or, for function parameters, at the end of the function).

In actuality, the C++ specification gives compilers a lot of flexibility to determine when local variables are created and destroyed. Objects may be created earlier, or destroyed later for optimization purposes. Most often, local variables are created when the function is entered, and destroyed in the opposite order of creation when the function is exited. This is the call stack.

If the object is a class type object, prior to destruction, a special function called a destructor is invoked. In many cases, the destructor does nothing, in which case no cost is incurred

Any use of an object after it has been destroyed will result in undefined behavior.

At some point after destruction, the memory used by the object will be **deallocated** (freed up for reuse).

Local scope (block scope)

An identifier’s **scope** determines where the identifier can be seen and used within the source code. When an identifier can be seen and used, we say it is **in scope**. When an identifier can not be seen, we can not use it, and we say it is **out of scope**. Scope is a compile-time property, and trying to use an identifier when it is not in scope will result in a compile error


An identifier is out of scope anywhere it cannot be accessed within the code. In the example above, the identifier `x` is in scope from its point of definition to the end of the `main` function. The identifier `x` is out of scope outside of that code region.

The term “going out of scope” is typically applied to objects rather than identifiers. We say an object goes out of scope at the end of the scope (the end curly brace) in which the object was instantiated. In the example above, the object named `x` goes out of scope at the end of the function `main`.


A local variable’s lifetime ends at the point where it goes out of scope, so local variables are destroyed at this point.

Note that not all types of variables are destroyed when they go out of scope. We’ll see examples of these in future lessons.

Key insight

Names used for function parameters or variables declared in a function body are only visible within the function that declares them. This means local variables within a function can be named without regard for the names of variables in other functions. This helps keep functions independent.


Best practice

Define your local variables as close to their first use as reasonable.

Introduction to temporary objects[](https://www.learncpp.com/cpp-tutorial/introduction-to-local-scope/#temporaries)

A **temporary object** (also sometimes called an **anonymous object**) is an unnamed object that is used to hold a value that is only needed for a short period of time. Temporary objects are generated by the compiler when they are needed.

Key insight

Return by value returns a temporary object (that holds a copy of the return value) to the caller.

Temporary objects have no scope at all (this makes sense, since scope is a property of an identifier, and temporary objects have no identifier).

Temporary objects are destroyed at the end of the full expression in which they are created. This means temporary objects are always destroyed before the next statement executes.


### Forward declarations and definitions


Best practice

When addressing compilation errors or warnings in your programs, resolve the first issue listed and then compile again.

A **forward declaration** allows us to tell the compiler about the existence of an identifier _before_ actually defining the identifier.

To write a forward declaration for a function, we use a **function declaration** statement (also called a **function prototype**). The function declaration consists of the function’s return type, name, and parameter types, terminated with a semicolon

Here’s a function declaration for the _add_ function:

```cpp
int add(int x, int y); // function declaration includes return type, name, parameters, and semicolon.  No function body!
```

Now when the compiler reaches the call to _add_ in main, it will know what _add_ looks like (a function that takes two integer parameters and returns an integer), and it won’t complain.

It is worth noting that function declarations do not need to specify the names of the parameters (as they are not considered to be part of the function declaration). In the above code, you can also forward declare your function like this:

```cpp
int add(int, int); // valid function declaration
```

Best practice

Keep the parameter names in your function declarations.


If a forward declaration is made, but the function is never called, the program will compile and run fine. However, if a forward declaration is made and the function is called, but the program never defines the function, the program will compile okay, but the linker will complain that it can’t resolve the function call.

Declarations vs. definitions

A **declaration** tells the _compiler_ about the _existence_ of an identifier and its associated type information. Here are some examples of declarations:

A **definition** is a declaration that actually implements (for functions and types) or instantiates (for variables) the identifier.

In C++, all definitions are declarations. Therefore `int x;` is both a definition and a declaration.

Conversely, not all declarations are definitions. Declarations that aren’t definitions are called **pure declarations**

In common language, the term “declaration” is typically used to mean “a pure declaration”, and “definition” is used to mean “a definition that also serves as a declaration”.

In most cases, a declaration is sufficient to allow the compiler to ensure an identifier is being used properly

However, there are a few cases where the compiler must be able to see a full definition in order to use an identifier (such as for template definitions and type definitions)


The one definition rule (ODR)[](https://www.learncpp.com/cpp-tutorial/forward-declarations/#ODR)

The **one definition rule** (or ODR for short) is a well-known rule in C++. The ODR has three parts:

1. Within a _file_, each function, variable, type, or template in a given scope can only have one definition. Definitions occurring in different scopes (e.g. local variables defined inside different functions, or functions defined inside different namespaces) do not violate this rule.
2. Within a _program_, each function or variable in a given scope can only have one definition. This rule exists because programs can have more than one file (we’ll cover this in the next lesson). Functions and variables not visible to the linker are excluded from this rule (discussed further in lesson [7.6 -- Internal linkage](https://www.learncpp.com/cpp-tutorial/internal-linkage/)).
3. Types, templates, inline functions, and inline variables are allowed to have duplicate definitions in different files, so long as each definition is identical. We haven’t covered what most of these things are yet, so don’t worry about this for now -- we’ll bring it back up when it’s relevant.


Violating part 1 of the ODR will cause the compiler to issue a redefinition error. Violating ODR part 2 will cause the linker to issue a redefinition error. Violating ODR part 3 will cause undefined behavior.

For advanced readers

Functions that share an identifier but have different sets of parameters are also considered to be distinct functions, so such definitions do not violate the ODR. We discuss this further in lesson [11.1 -- Introduction to function overloading](https://www.learncpp.com/cpp-tutorial/introduction-to-function-overloading/).

Functions that share an identifier but have different sets of parameters are also considered to be distinct functions, so such definitions do not violate the ODR. We discuss this further in lesson [11.1 -- Introduction to function overloading](https://www.learncpp.com/cpp-tutorial/introduction-to-function-overloading/).


For VS Code users

To create a new file, choose _View > Explorer_ from the top nav to open the Explorer pane, and then click the _New File_ icon to the right of the project name. Alternately, choose _File > New File_ from the top nav. Then give your new file a name (don’t forget the .cpp extension). If the file appears inside the _.vscode_ folder, drag it up one level to the project folder.

Next open the _tasks.json_ file, and find the line `"${file}",`.

You have two options here:

- If you wish to be explicit about what files get compiled, replace `"${file}",` with the name of each file you wish to compile, one per line, like this:

`"main.cpp",`  
`"add.cpp",`

- Reader “geo” reports that you can have VS Code automatically compile all .cpp files in the directory by replacing `"${file}",` with `"${fileDirname}\\**.cpp"` (on Windows).
- Reader “Orb” reports that `"${fileDirname}/**.cpp"` works on Unix.


the compiler compiles each file individually. It does not know about the contents of other code files, or remember anything it has seen from previously compiled code files. So even though the compiler may have seen the definition of function _add_ previously (if it compiled _add.cpp_ first), it doesn’t remember.

This limited visibility and short memory is intentional, for a few reasons:

1. It allows the source files of a project to be compiled in any order.
2. When we change a source file, only that source file needs to be recompiled.
3. It reduces the possibility of naming conflicts between identifiers in different files.

When we use forward declarations: when the compiler is compiling _main.cpp_, it will know what identifier _add_ is and be satisfied. The linker will connect the function call to _add_ in _main.cpp_ to the definition of function _add_ in _add.cpp_.


### Naming collisions and an introduction to namespaces


Similarly, C++ requires that all identifiers be non-ambiguous. If two identical identifiers are introduced into the same program in a way that the compiler or linker can’t tell them apart, the compiler or linker will produce an error. This error is generally referred to as a **naming collision** (or **naming conflict**).

If the colliding identifiers are introduced into the same file, the result will be a compiler error. If the colliding identifiers are introduced into separate files belonging to the same program, the result will be a linker error.

However, when the linker executes, it will link all the definitions in _a.cpp_ and _main.cpp_ together, and discover conflicting definitions for function `myFcn()`. The linker will then abort with an error. Note that this error occurs even though `myFcn()` is never called!


A **scope region** is an area of source code where all declared identifiers are considered distinct from names declared in other scopes (much like the cities in our analogy). Two identifiers with the same name can be declared in separate scope regions without causing a naming conflict


The body of a function is one example of a scope region. Two identically-named identifiers can be defined in separate functions without issue -- because each function provides a separate scope region, there is no collision. However, if you try to define two identically-named identifiers within the same function, a naming collision will result, and the compiler will complain.

Namespaces

A **namespace** provides another type of scope region (called **namespace scope**) that allows you to declare or define names inside of it for the purpose of disambiguation. The names declared in a namespace are isolated from names declared in other scopes, allowing such names to exist without conflict.


Namespaces may only contain declarations and definitions (e.g. variables and functions). Executable statements are not allowed unless they are part of a definition (e.g. within a function).
A namespace may only contain declarations and definitions. Executable statements are only allowed as part of a definition (e.g. of a function).



The global namespace

In C++, any name that is not defined inside a class, function, or a namespace is considered to be part of an implicitly-defined namespace called the **global namespace** (sometimes also called **the global scope**).

- Identifiers declared inside the global scope are in scope from the point of declaration to the end of the file.
- Although variables can be defined in the global namespace, this should generally be avoided (we discuss why in lesson [7.8 -- Why (non-const) global variables are evil](https://www.learncpp.com/cpp-tutorial/why-non-const-global-variables-are-evil/)).




```cpp
#include <iostream> // imports the declaration of std::cout into the global scope

// All of the following statements are part of the global namespace

void foo();    // okay: function forward declaration
int x;         // compiles but strongly discouraged: non-const global variable definition (without initializer)
int y { 5 };   // compiles but strongly discouraged: non-const global variable definition (with initializer)
x = 5;         // compile error: executable statements are not allowed in namespaces

int main()     // okay: function definition
{
    return 0;
}

void goo();    // okay: A function forward declaration
```

When you use an identifier that is defined inside a non-global namespace (e.g. the `std` namespace), you need to tell the compiler that the identifier lives inside the namespace.”



The :: symbol is an operator called the **scope resolution operator**. The identifier to the left of the `::` symbol identifies the namespace that the name to the right of the `::` symbol is contained within. If no identifier to the left of the `::` symbol is provided, the global namespace is assumed.

Use explicit namespace prefixes to access identifiers defined in a namespace.

Another way to access identifiers inside a namespace is to use a using-directive statement. Here’s our original “Hello world” program with a using-directive: 
A **using directive** allows us to access the names in a namespace without using a namespace prefix. So in the above example, when the compiler goes to determine what identifier `cout` is, it will match with `std::cout`, which, because of the using-directive, is accessible as just `cout`.

Many texts, tutorials, and even some IDEs recommend or use a using-directive at the top of the program. However, used in this way, this is a bad practice, and highly discouraged.

When using a using-directive in this manner, _any_ identifier we define may conflict with _any_ identically named identifier in the `std` namespace. Even worse, while an identifier name may not conflict today, it may conflict with new identifiers added to the std namespace in future language revisions. This was the whole point of moving all of the identifiers in the standard library into the `std` namespace in the first place!


Warning

Avoid using-directives (such as `using namespace std;`) at the top of your program or in header files. They violate the reason why namespaces were added in the first place.


In C++, curly braces are often used to delineate a scope region that is nested within another scope region (braces are also used for some non-scope-related purposes, such as list initialization). For example, a function defined inside the global scope region uses curly braces to separate the scope region of the function from the global scope.


### Introduction to the preprocessor

Instead, prior to compilation, each code (.cpp) file goes through a **preprocessing** phase. In this phase, a program called the **preprocessor** makes various changes to the text of the code file. The preprocessor does not actually modify the original code files in any way -- rather, all changes made by the preprocessor happen either temporarily in-memory or using temporary files.

Most of what the preprocessor does is fairly uninteresting. For example, it strips out comments, and ensures each code file ends in a newline. However, the preprocessor does have one very important role: it is what processes `#include` directives (which we’ll discuss more in a moment).

When the preprocessor has finished processing a code file, the result is called a **translation unit**. This translation unit is what is then compiled by the compiler.

The entire process of preprocessing, compiling, and linking is called **translation**.


Preprocessor directives

When the preprocessor runs, it scans through the code file (from top to bottom), looking for preprocessor directives. **Preprocessor directives** (often just called _directives_) are instructions that start with a _#_ symbol and end with a newline (NOT a semicolon). These directives tell the preprocessor to perform certain text manipulation tasks. Note that the preprocessor does not understand C++ syntax -- instead, the directives have their own syntax (which in some cases resembles C++ syntax, and in other cases, not so much).


The final output of the preprocessor contains no directives -- only the output of the processed directive is passed to the compiler.


You’ve already seen the _#include_ directive in action (generally to #include \<iostream>). When you _#include_ a file, the preprocessor replaces the #include directive with the contents of the included file. The included contents are then preprocessed (which may result in additional #includes being preprocessed recursively), then the rest of the file is preprocessed.


When the preprocessor runs on this program, the preprocessor will replace `#include <iostream>` with the contents of the file named “iostream” and then preprocess the included content and the rest of the file.

Macros

The _#define_ directive can be used to create a macro. In C++, a **macro** is a rule that defines how input text is converted into replacement output text.

There are two basic types of macros: _object-like macros_, and _function-like macros_.

_Function-like macros_ act like functions, and serve a similar purpose. Their use is generally considered unsafe, and almost anything they can do can be done by a normal function.

_Object-like macros_ can be defined in one of two ways:

#define IDENTIFIER
#define IDENTIFIER substitution_text

The top definition has no substitution text, whereas the bottom one does. Because these are preprocessor directives (not statements), note that neither form ends with a semicolon.

Object-like macros with substitution text were used (in C) as a way to assign names to literals. This is no longer necessary, as better methods are available in C++ (see [7.10 -- Sharing global constants across multiple files (using inline variables)](https://www.learncpp.com/cpp-tutorial/sharing-global-constants-across-multiple-files-using-inline-variables/)). Object-like macros with substitution text are now mostly seen in legacy code, and we recommend avoiding them whenever possible.


_Object-like macros_ can also be defined without substitution text.

For example:

```cpp
#define USE_YEN
```

Macros of this form work like you might expect: most further occurrences of the identifier is removed and replaced by nothing!

This might seem pretty useless, and it _is useless_ for doing text substitution. However, that’s not what this form of the directive is generally used for. We’ll discuss the uses of this form in just a moment.

Unlike object-like macros with substitution text, macros of this form are generally considered acceptable to use.


Conditional compilation

The _conditional compilation_ preprocessor directives allow you to specify under what conditions something will or won’t compile. There are quite a few different conditional compilation directives, but we’ll only cover a few that are used the most often: _#ifdef_, _#ifndef_, and _#endif_.

The _#ifdef_ preprocessor directive allows the preprocessor to check whether an identifier has been previously defined via #define. If so, the code between the _#ifdef_ and matching _#endif_ is compiled. If not, the code is ignored.

Consider the following program:

```cpp
#include <iostream>

#define PRINT_JOE

int main()
{
#ifdef PRINT_JOE
    std::cout << "Joe\n"; // will be compiled since PRINT_JOE is defined
#endif

#ifdef PRINT_BOB
    std::cout << "Bob\n"; // will be excluded since PRINT_BOB is not defined
#endif

    return 0;
}
```


_#ifndef_ is the opposite of _#ifdef_, in that it allows you to check whether an identifier has _NOT_ been _#define_d yet.

In place of `#ifdef PRINT_BOB` and `#ifndef PRINT_BOB`, you’ll also see `#if defined(PRINT_BOB)` and `#if !defined(PRINT_BOB)`. These do the same, but use a slightly more C++-style syntax.

#if 0[](https://www.learncpp.com/cpp-tutorial/introduction-to-the-preprocessor/#if0)

One more common use of conditional compilation involves using _#if 0_ to exclude a block of code from being compiled (as if it were inside a comment block):

```cpp
#include <iostream>

int main()
{
    std::cout << "Joe\n";

#if 0 // Don't compile anything starting here
    std::cout << "Bob\n";
    std::cout << "Steve\n";
#endif // until this point

    return 0;
}
```

This provides a convenient way to “comment out” code that contains multi-line comments (which can’t be commented out using another multi-line comment due to multi-line comments being non-nestable):

Macro substitution within other preprocessor commands

Now you might be wondering, given the following code:

```cpp
#define PRINT_JOE

int main()
{
#ifdef PRINT_JOE
    std::cout << "Joe\n"; // will be compiled since PRINT_JOE is defined
#endif

    return 0;
}
```

Since we defined _PRINT_JOE_ to be nothing, how come the preprocessor didn’t replace _PRINT_JOE_ in _#ifdef PRINT_JOE_ with nothing and exclude the output statement from compilation?

In most cases, macro substitution does not occur when a macro identifier is used within another preprocessor command.

The scope of #defines

Directives are resolved before compilation, from top to bottom on a file-by-file basis.

Consider the following program:

```cpp
#include <iostream>

void foo()
{
#define MY_NAME "Alex"
}

int main()
{
	std::cout << "My name is: " << MY_NAME << '\n';

	return 0;
}
```

Even though it looks like _#define MY_NAME “Alex”_ is defined inside function _foo_, the preprocessor doesn’t understand C++ concepts like functions. Therefore, this program behaves identically to one where _#define MY_NAME “Alex”_ was defined either before or immediately after function _foo_. To avoid confusion, you’ll generally want to #define identifiers outside of functions.


Once the preprocessor has finished, all defined identifiers from that file are discarded. This means that directives are only valid from the point of definition to the end of the file in which they are defined. Directives defined in one file do not have any impact on other files (unless they are #included into another file). For example:


function.cpp:

```cpp
#include <iostream>

void doSomething()
{
#ifdef PRINT
    std::cout << "Printing!\n";
#endif
#ifndef PRINT
    std::cout << "Not printing!\n";
#endif
}
```

main.cpp:

```cpp
void doSomething(); // forward declaration for function doSomething()

#define PRINT

int main()
{
    doSomething();

    return 0;
}
```

The above program will print:

Not printing!



### Header files:

When you `#include` a file, the content of the included file is inserted at the point of inclusion. This provides a useful way to pull in declarations from another file.

If using the command line, just create a new file in your favorite editor in the same directory as your source (.cpp) files. Unlike source files, header files should _not_ be added to your compile command (they are implicitly included by #include statements and compiled as part of your source files).

Prefer a .h suffix when naming your header files (unless your project already follows some other convention).

This is a longstanding convention for C++ header files, and most IDEs still default to .h over other options.

Header files are often paired with code files, with the header file providing forward declarations for the corresponding code file. Since our header file will contain a forward declaration for functions defined in _add.cpp_, we’ll call our new header file _add.h_.

Best practice

If a header file is paired with a code file (e.g. add.h with add.cpp), they should both have the same base name (add).

In order to use this header file in main.cpp, we have to #include it (using quotes, not angle brackets).

main.cpp:

```cpp
#include "add.h" // Insert contents of add.h at this point.  Note use of double quotes here.
#include <iostream>

int main()
{
    std::cout << "The sum of 3 and 4 is " << add(3, 4) << '\n';
    return 0;
}
```

add.cpp:

```cpp
#include "add.h" // Insert contents of add.h at this point.  Note use of double quotes here.

int add(int x, int y)
{
    return x + y;
}
```


Source files should include their paired header[](https://www.learncpp.com/cpp-tutorial/header-files/#corresponding_include)

In C++, it is a best practice for code files to #include their paired header file (if one exists). This allows the compiler to catch certain kinds of errors at compile time instead of link time. 
Because if you change definiton of a function in the source code, and you forget to change its header, then it will give a compile error instead of linking error, otherwise at link time it will see two funcs with diff definitions

As an aside…

Unfortunately, this doesn’t work if it is a parameter with a different type instead of a return type. This is because C++ supports overloaded functions (functions with the same name but different parameter types), so the compiler will assume a function with a mismatched parameter type is a different overload. Can’t win em all. What are overloaded functions?

We will also see many examples in future lessons where content required by the source file is defined in the paired header. In such cases, including the header is a necessity.

Best practice

Source files should #include their paired header file (if one exists).


How including definitions in a header file results in a violation of the one-definition rule

For now, you should avoid putting function or variable definitions in header files. Doing so will generally result in a violation of the one-definition rule (ODR) in cases where the header file is included into more than one source file.



Best practice

Do not put function and variable definitions in your header files (for now).

Defining either of these in a header file will likely result in a violation of the one-definition rule (ODR) if that header is then #included into more than one source (.cpp) file.

Author’s note

In future lessons, we will encounter additional kinds of definitions that can be safely defined in header files (because they are exempt from the ODR). This includes definitions for inline functions, inline variables, types, and templates. We’ll discuss this further when we introduce each of these.



Do not #include .cpp files[](https://www.learncpp.com/cpp-tutorial/header-files/#includecpp)

Although the preprocessor will happily do so, you should generally not `#include` .cpp files. These should be added to your project and compiled.

There are number of reasons for this:

- Doing so can cause naming collisions between source files.
- In a large project it can be hard to avoid one definition rules (ODR) issues.
- Any change to such a .cpp file will cause both the .cpp file and any other .cpp file that includes it to recompile, which can take a long time. Headers tend to change less often than source files.
- It is non-conventional to do so.

When we use angled brackets, we’re telling the preprocessor that this is a header file we didn’t write ourselves. The preprocessor will search for the header only in the directories specified by the `include directories`. The `include directories` are configured as part of your project/IDE settings/compiler settings, and typically default to the directories containing the header files that come with your compiler and/or OS. The preprocessor will not search for the header file in your project’s source code directory.


Use double quotes to include header files that you’ve written or are expected to be found in the current directory. Use angled brackets to include headers that come with your compiler, OS, or third-party libraries you’ve installed elsewhere on your system.


When C++ was first created, all of the headers in the standard library ended in a _.h_ suffix. These headers included:

|Header type|Naming convention|Example|Identifiers placed in namespace|
|---|---|---|---|
|C++ specific|<xxx.h>|iostream.h|Global namespace|
|C compatibility|<xxx.h>|stddef.h|Global namespace|

The original versions of _cout_ and _cin_ were declared in _iostream.h_ in the global namespace. Life was consistent, and it was good.

When the language was standardized by the ANSI committee, they decided to move all of the names used in the standard library into the _std_ namespace to help avoid naming conflicts with user-declared identifiers. However, this presented a problem: if they moved all the names into the _std_ namespace, none of the old programs (that included iostream.h) would work anymore!

To work around this issue, C++ introduced new header files that lack the _.h_ extension. These new header files declare all names inside the _std_ namespace. This way, older programs that include `#include <iostream.h>` do not need to be rewritten, and newer programs can `#include <iostream>`.

Modern C++ now contains 4 sets of header files:

|Header type|Naming convention|Example|Identifiers placed in namespace|
|---|---|---|---|
|C++ specific (new)|<xxx>|iostream|`std` namespace|
|C compatibility (new)|<cxxx>|cstddef|`std` namespace (required)  <br>global namespace (optional)|
|C++ specific (old)|<xxx.h>|iostream.h|Global namespace|
|C compatibility (old)|<xxx.h>|stddef.h|Global namespace (required)  <br>`std` namespace (optional)|

Warning

The new C compatibility headers \<cxxx> may optionally declare names in the global namespace, and the old C compatibility headers <xxx.h> may optionally declare names in the `std` namespace. Names in these locations should be avoided, as those names may not be declared in those locations on other implementations.

Best practice

Use the standard library header files without the .h extension. User-defined headers should still use a .h extension.



Including header files from other directories

Another common question involves how to include header files from other directories.

One (bad) way to do this is to include a relative path to the header file you want to include as part of the #include line. For example:

```cpp
#include "headers/myHeader.h"
#include "../moreHeaders/myOtherHeader.h"
```

While this will compile (assuming the files exist in those relative directories), the downside of this approach is that it requires you to reflect your directory structure in your code. If you ever update your directory structure, your code won’t work anymore.

A better method is to tell your compiler or IDE that you have a bunch of header files in some other location, so that it will look there when it can’t find them in the current directory. This can generally be done by setting an _include path_ or _search directory_ in your IDE project settings.


For gcc users

Using g++, you can use the -I option to specify an alternate include directory:  
`g++ -o main -I./source/includes main.cpp`

There is no space after the `-I`. For a full path (rather than a relative path), remove the `.` after `-I`.

For VS Code users

In your _tasks.json_ configuration file, add a new line in the _“Args”_ section:  
`"-I./source/includes",`

There is no space after the `-I`. For a full path (rather than a relative path), remove the `.` after `-I`.

The nice thing about this approach is that if you ever change your directory structure, you only have to change a single compiler or IDE setting instead of every code file.


It’s common that the content of a header file will make use of something that is declared (or defined) in another header file. When this happens, the header file should #include the other header file containing the declaration (or definition) that it needs.

Foo.h:

```cpp
#include \<string_view> // required to use std::string_view

std::string_view getApplicationName(); // std::string_view used here
```

Transitive includes

When your source (.cpp) file #includes a header file, you’ll also get any other header files that are #included by that header (and any header files those include, and so on). These additional header files are sometimes called **transitive includes**, as they’re included implicitly rather than explicitly.


Best practice

Each file should explicitly #include all of the header files it needs to compile. Do not rely on headers included transitively from other headers.

The order of inclusion for header files

If your header files are written properly and #include everything they need, the order of inclusion shouldn’t matter.

Now consider the following scenario: let’s say header A needs declarations from header B, but forgets to include it. In our code file, if we include header B before header A, our code will still compile! This is because the compiler will compile all the declarations from B before it compiles the code from A that depends on those declarations.


Best practice

To maximize the chance that missing includes will be flagged by compiler, order your #includes as follows (skipping any that are not relevant):

- The paired header file for this code file (e.g. `add.cpp` should `#include "add.h"`)
- Other headers from the same project (e.g. `#include "mymath.h"`)
- 3rd party library headers (e.g. `#include \<boost/tuple/tuple.hpp>`)
- Standard library headers (e.g. `#include \<iostream>`)

The headers for each grouping should be sorted alphabetically (unless the documentation for a 3rd party library instructs you to do otherwise).

That way, if one of your user-defined headers is missing an #include for a 3rd party library or standard library header, it’s more likely to cause a compile error so you can fix it.


Duplicate definitions and a compile error. Each file, individually, is fine. However, because _main.cpp_ ends up #including the content of _square.h_ twice, we’ve run into problems. If _wave.h_ needs _getSquareSides()_, and _main.cpp_ needs both _wave.h_ and _square.h_, how would you resolve this issue?

Header guards

The good news is that we can avoid the above problem via a mechanism called a **header guard** (also called an **include guard**). Header guards are conditional compilation directives that take the following form:


```cpp
#ifndef SOME_UNIQUE_NAME_HERE
#define SOME_UNIQUE_NAME_HERE

// your declarations (and certain types of definitions) here

#endif
```

When this header is #included, the preprocessor will check whether _SOME_UNIQUE_NAME_HERE_ has been previously defined in this translation unit. If this is the first time we’re including the header, _SOME_UNIQUE_NAME_HERE_ will not have been defined. Consequently, it #defines _SOME_UNIQUE_NAME_HERE_ and includes the contents of the file. If the header is included again into the same file, _SOME_UNIQUE_NAME_HERE_ will already have been defined from the first time the contents of the header were included, and the contents of the header will be ignored (thanks to the #ifndef).


All of your header files should have header guards on them. _SOME_UNIQUE_NAME_HERE_ can be any name you want, but by convention is set to the full filename of the header file, typed in all caps, using underscores for spaces or punctuation. For example, _square.h_ would have the header guard:

square.h:

```cpp
#ifndef SQUARE_H
#define SQUARE_H

int getSquareSides()
{
    return 4;
}

#endif
```


Even the standard library headers use header guards. If you were to take a look at the iostream header file from Visual Studio, you would see:

```cpp
#ifndef _IOSTREAM_
#define _IOSTREAM_

// content here

#endif
```

For advanced readers

In large programs, it’s possible to have two separate header files (included from different directories) that end up having the same filename (e.g. directoryA\config.h and directoryB\config.h). If only the filename is used for the include guard (e.g. CONFIG_H), these two files may end up using the same guard name. If that happens, any file that includes (directly or indirectly) both config.h files will not receive the contents of the include file to be included second. This will probably cause a compilation error.

Because of this possibility for guard name conflicts, many developers recommend using a more complex/unique name in your header guards. Some good suggestions are a naming convention of PROJECT_PATH_FILE_H, FILE_LARGE-RANDOM-NUMBER_H, or FILE_CREATION-DATE_H.

Updating our previous example with header guards

Let’s return to the _square.h_ example, using the _square.h_ with header guards. For good form, we’ll also add header guards to _wave.h_.

square.h

```cpp
#ifndef SQUARE_H
#define SQUARE_H

int getSquareSides()
{
    return 4;
}

#endif
```

wave.h:

```cpp
#ifndef WAVE_H
#define WAVE_H

#include "square.h"

#endif
```

main.cpp:

```cpp
#include "square.h"
#include "wave.h"

int main()
{
    return 0;
}
```

After the preprocessor resolves all of the #include directives, this program looks like this:

main.cpp:

```cpp
// Square.h included from main.cpp
#ifndef SQUARE_H // square.h included from main.cpp
#define SQUARE_H // SQUARE_H gets defined here

// and all this content gets included
int getSquareSides()
{
    return 4;
}

#endif // SQUARE_H

#ifndef WAVE_H // wave.h included from main.cpp
#define WAVE_H
#ifndef SQUARE_H // square.h included from wave.h, SQUARE_H is already defined from above
#define SQUARE_H // so none of this content gets included

int getSquareSides()
{
    return 4;
}

#endif // SQUARE_H
#endif // WAVE_H

int main()
{
    return 0;
}
```


Header guards do not prevent a header from being included once into different code files

Note that the goal of header guards is to prevent a code file from receiving more than one copy of a guarded header. By design, header guards do _not_ prevent a given header file from being included (once) into separate code files. This can also cause unexpected problems. Consider:


Can’t we just avoid definitions in header files?

We’ve generally told you not to include function definitions in your headers. So you may be wondering why you should include header guards if they protect you from something you shouldn’t do.

There are quite a few cases we’ll show you in the future where it’s necessary to put non-function definitions in a header file. For example, C++ will let you create your own types. These custom types are typically defined in header files, so the type definitions can be propagated out to the code files that need to use them. Without a header guard, a code file could end up with multiple (identical) copies of a given type definition, which the compiler will flag as an error.

So even though it’s not strictly necessary to have header guards at this point in the tutorial series, we’re establishing good habits now, so you don’t have to unlearn bad habits later.


#pragma once

Modern compilers support a simpler, alternate form of header guards using the `#pragma` preprocessor directive:

```cpp
#pragma once

// your code here
```

`#pragma once` serves the same purpose as header guards: to avoid a header file from being included multiple times. With traditional header guards, the developer is responsible for guarding the header (by using preprocessor directives `#ifndef`, `#define`, and `#endif`). With `#pragma once`, we’re requesting that the compiler guard the header. How exactly it does this is an implementation-specific detail.

For advanced readers

There is one known case where `#pragma once` will typically fail. If a header file is copied so that it exists in multiple places on the file system, if somehow both copies of the header get included, header guards will successfully de-dupe the identical headers, but `#pragma once` won’t (because the compiler won’t realize they are actually identical content).

Because pragma once probably generates a unique id each time, and for diff files it will create diff unique values hence in the eyes of the compiler it would be ok


For most projects, `#pragma once` works fine, and many developers now prefer it because it is easier and less error-prone. Many IDEs will also auto-include `#pragma once` at the top of a new header file generated through the IDE.

Warning

The `#pragma` directive was designed for compiler implementers to use for whatever purposes they desire. As such, which pragmas are supported and what meaning those pragmas have is completely implementation-specific. With the exception of `#pragma once`, do not expect a pragma that works on one compiler to be supported by another.


Because `#pragma once` is not defined by the C++ standard, it is possible that some compilers may not implement it. For this reason, some development houses (such as Google) recommend using traditional header guards. In this tutorial series, we will favor header guards, as they are the most conventional way to guard headers. However, support for `#pragma once` is fairly ubiquitous at this point, and if you wish to use `#pragma once` instead, that is generally accepted in modern C++.


