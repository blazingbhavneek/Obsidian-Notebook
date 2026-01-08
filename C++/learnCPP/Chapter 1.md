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


You’ve already seen the _#include_ directive in action (generally to #include <iostream>). When you _#include_ a file, the preprocessor replaces the #include directive with the contents of the included file. The included contents are then preprocessed (which may result in additional #includes being preprocessed recursively), then the rest of the file is preprocessed.


When the preprocessor runs on this program, the preprocessor will replace `#include <iostream>` with the contents of the file named “iostream” and then preprocess the included content and the rest of the file.


