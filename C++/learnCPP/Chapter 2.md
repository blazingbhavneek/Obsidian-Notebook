
## Functions

### Basic Terminology

- **Function**: Reusable sequence of statements designed to do a particular job
- **Caller**: Function initiating the function call
- **Callee**: Function being called/executed
- **Invocation**: Another term for function call
- **Function header**: First line of function (informal term) - tells compiler about function's existence, name, and other information
- **Metasyntactic variables**: Meaningless placeholder names (like "foo") used when the actual name is unimportant to demonstration

### Variable Initialization Syntax

- Can initialize variables with curly braces: `int num { getValueFromUser() };`

### Main Function Special Cases

#### Execution Order

- Common misconception: `main` is NOT always the first function that executes
- Global variables are initialized prior to `main()` execution
- If a global variable's initializer invokes a function, that function executes before `main`

#### Calling Main

- C++ standard prohibits calling `main()` explicitly
- Some C++ compilers allow this for C compatibility reasons (C allows it)

#### Return Values (Status Codes)

- **Status code** (also: exit code, return code): Return value from `main()`
- Signals whether program was successful
- Standard defines only 3 status codes:
    - `0`: Program executed successfully
    - `EXIT_SUCCESS`: Program executed successfully (same as 0)
    - `EXIT_FAILURE`: Program did not execute successfully
- `EXIT_SUCCESS` and `EXIT_FAILURE` are preprocessor macros in `<cstdlib>`
- **Best practice**: Use only `0`/`EXIT_SUCCESS` for success or `EXIT_FAILURE` for failure to maximize portability
- **Special case**: `main()` implicitly returns 0 if no return statement provided

## Return Values

### Value-Returning Functions

- Must return a value or produce undefined behavior
- Compilers usually detect missing returns, but may miss some in complicated cases
- Can only return a single value per call
- Return value doesn't need to be literal - can be expression, variable, or another function call
- Various workarounds exist for returning multiple values (covered in future lessons)

### Non-Value Returning Functions

- **Best practice**: Do NOT put return statement at end of non-value returning function

### Function Calls as Statements

- When calling for behavior (not return value), can call either:
    - Non-value returning function
    - Value-returning function and ignore return value

## Function Parameters

### Unreferenced and Unnamed Parameters

* Compiler warns about defined but unused parameters
* **Unnamed parameter**: Parameter without a name in function definition
* Used when a parameter must exist for the function signature but isn’t used in the function body

#### Common Scenarios for Unnamed Parameters

1. **Function updated but parameter no longer needed**
   Removing the parameter would break existing callers, so the parameter stays but is unused.

2. **Cannot change all calling code**
   Common in libraries or callbacks where you don’t control the caller.

3. **Operator overloading (prefix vs postfix)**
   C++ needs different function signatures to distinguish `++foo` from `foo++`.

   * Prefix increment:

     ```cpp
     T& operator++();
     ```

     No parameters.

   * Postfix increment:

     ```cpp
     T operator++(int);
     ```

     The `int` parameter is a **dummy parameter**.

     * It is never used.
     * The caller never provides it.
     * It exists only so the compiler can tell postfix apart from prefix.

   Best practice:

   ```cpp
   T operator++(int /*unused*/);
   ```

4. **Type template parameters (type matters, value doesn’t)**
   Sometimes a parameter exists only so the compiler can deduce a template type.

   Example:

   ```cpp
   template <typename T>
   void f(T) {
       // logic depends on T, not the value
   }
   ```

   * The argument’s value is irrelevant.
   * Only the type `T` is used at compile time.

   If the value isn’t needed:

   ```cpp
   template <typename T>
   void f(T /*unused*/) { }
   ```

   The parameter exists purely as a **type carrier**.

**Best practice**
When a parameter must exist but is unused, don’t give it a name.
Optionally document it with a comment.


## Variable Lifetime & Scope

### Lifetime

- **Lifetime**: Time between variable creation and destruction
- Runtime property (not compile-time)
- Creation and destruction happen during program execution

#### Guarantees

- Objects must be created/initialized no later than point of definition
- Must be destroyed no earlier than end of curly braces where defined (or end of function for parameters)
- Compiler has flexibility: may create earlier or destroy later for optimization
- Usually: local variables created when function entered, destroyed in reverse order when function exits (call stack)

#### Destruction

- Class objects invoke destructor before destruction (often does nothing)
- Using object after destruction = undefined behavior
- After destruction, memory is deallocated (freed for reuse)

### Scope

- **Scope**: Determines where identifier can be seen/used in source code
- **In scope**: Identifier can be seen and used
- **Out of scope**: Identifier cannot be seen or used
- Compile-time property (not runtime)
- Using out-of-scope identifier = compile error

#### Local/Block Scope

- Identifier in scope from point of definition to end of block (closing curly brace)
- "Going out of scope": Applied to objects, happens at end curly brace of instantiation scope
- Local variable lifetime ends when it goes out of scope → destroyed at this point
- **Exception**: Not all variable types are destroyed when going out of scope

**Key insight**: Function parameters and local variables only visible within declaring function. Enables naming without regard to other functions' variables. Keeps functions independent.

**Best practice**: Define local variables as close to first use as reasonable.

### Temporary Objects

- **Temporary object** (anonymous object): Unnamed object holding value needed briefly
- Generated by compiler when needed
- **Key insight**: Return by value returns temporary object (copy of return value) to caller
- Have no scope (scope is identifier property, temporaries have no identifier)
- Destroyed at end of full expression where created (always before next statement)

## Forward Declarations & Definitions

### Compilation Errors

**Best practice**: When addressing errors/warnings, resolve first issue listed, then recompile.

### Forward Declaration

- Tells compiler about identifier existence before defining it
- **Function declaration** (function prototype): Forward declaration for function
- Contains: return type, name, parameter types, semicolon
- **No function body**
- Example: `int add(int x, int y);`

#### Parameter Names in Declarations

- Not required in declarations (not part of declaration)
- Can declare as: `int add(int, int);`
- **Best practice**: Keep parameter names in declarations

#### Behavior

- If forward declared but never called: compiles and runs fine
- If forward declared, called, but never defined: compiles fine, linker complains (can't resolve call)

### Declarations vs Definitions

#### Declaration

- Tells compiler about identifier existence and type information
- Examples of declarations only:
    - `int add(int x, int y);` (function declaration)
    - `int x;` (variable declaration that's also a definition)

#### Definition

- Declaration that implements (functions/types) or instantiates (variables) the identifier
- **All definitions are declarations** (e.g., `int x;` is both)
- **Not all declarations are definitions**
- **Pure declaration**: Declaration that isn't a definition

#### Common Usage

- "Declaration" typically means "pure declaration"
- "Definition" means "definition that also serves as declaration"

#### When Definition Required

- Usually declaration sufficient for compiler to ensure proper usage
- Full definition required for:
    - Template definitions
    - Type definitions

## One Definition Rule (ODR)

### Three Parts

1. **Within a file**: Each function, variable, type, or template in given scope can only have one definition
    
    - Definitions in different scopes don't violate this (e.g., local variables in different functions)
2. **Within a program**: Each function or variable in given scope can only have one definition
    
    - Applies across multiple files
    - Functions/variables not visible to linker are excluded
3. **Types, templates, inline functions, inline variables**: Can have duplicate definitions in different files IF each definition is identical
    

### Violations

- Part 1 violation: Compiler issues redefinition error
- Part 2 violation: Linker issues redefinition error
- Part 3 violation: Undefined behavior

### Function Overloading Exception

- Functions with same identifier but different parameter sets are distinct functions
- Don't violate ODR

## Multi-File Projects

### VS Code Configuration

- Create new file: View > Explorer, click New File icon (or File > New File)
- Give file name with `.cpp` extension
- If appears in `.vscode` folder, drag to project folder

#### Compiling Multiple Files (tasks.json)

Replace `"${file}",` with either:

- **Explicit approach**: List each file
    
    ```
    "main.cpp","add.cpp",
    ```
    
- **Automatic approach**:
    - Windows: `"${fileDirname}\\**.cpp"`
    - Unix: `"${fileDirname}/**.cpp"`

### Compilation Process

- Compiler compiles each file individually
- Doesn't know about other code files
- Doesn't remember previously compiled files
- Even if saw function definition before, won't remember

#### Why Limited Visibility?

1. Allows files to be compiled in any order
2. When file changes, only that file needs recompilation
3. Reduces naming conflicts between files

#### Solution: Forward Declarations

- Compiler sees forward declaration, is satisfied
- Linker connects function call to definition in other file

## Naming Collisions & Namespaces

### Naming Collision (Naming Conflict)

- C++ requires all identifiers be non-ambiguous
- **Naming collision**: Two identical identifiers introduced so compiler/linker can't distinguish
- Same file: Compiler error
- Different files in same program: Linker error
- **Important**: Error occurs even if function never called

### Scope Regions

- **Scope region**: Area where declared identifiers are distinct from other scopes
- Like cities with same street names in different cities
- Examples: Different function bodies can have same variable names

### Namespaces

- Provides **namespace scope** for disambiguation
- Names declared in namespace isolated from other scopes
- Allows same names to exist without conflict

#### Namespace Rules

- May only contain declarations and definitions (variables, functions)
- Executable statements NOT allowed (unless part of definition like function body)

### Global Namespace (Global Scope)

- Any name not defined inside class, function, or namespace is in global namespace
- Implicitly defined
- Identifiers in scope from declaration to end of file
- **Strong recommendation**: Avoid defining variables in global namespace

#### Example of Global Namespace Content

```cpp
#include <iostream> // imports std::cout into global scope

void foo();         // OK: function forward declaration
int x;              // Compiles but discouraged: non-const global variable (no initializer)
int y { 5 };        // Compiles but discouraged: non-const global variable (with initializer)
x = 5;              // ERROR: executable statements not allowed in namespaces

int main()          // OK: function definition
{
    return 0;
}

void goo();         // OK: function forward declaration
```

### Accessing Namespace Content

#### Scope Resolution Operator (::)

- `::` is the scope resolution operator
- Identifier left of `::` = namespace name
- Identifier right of `::` = name within namespace
- No identifier left of `::` = global namespace assumed
- **Best practice**: Use explicit namespace prefixes to access namespace identifiers

#### Using Directive

- **Using directive**: Allows accessing namespace names without prefix
- Example: `using namespace std;`
- Compiler matches `cout` with `std::cout` automatically

##### Why Using Directives Are Bad (at top of program)

- ANY identifier you define may conflict with ANY identically named identifier in namespace
- May conflict with identifiers added to namespace in future language revisions
- Defeats the purpose of namespaces

**Warning**: Avoid using-directives (like `using namespace std;`) at top of program or in header files. They violate why namespaces were added.

### Nested Scope Regions

- Curly braces often delineate scope region nested within another
- Also used for non-scope purposes (like list initialization)
- Example: Function defined in global scope uses braces to separate function scope from global scope

## Preprocessor

### Overview

- Before compilation, code files go through **preprocessing phase**
- **Preprocessor**: Program that makes changes to code file text
- Doesn't modify original files - changes are temporary (in-memory or temporary files)

### Preprocessor Actions

- Strips out comments
- Ensures each file ends in newline
- **Most important**: Processes `#include` directives

### Translation Unit & Translation

- **Translation unit**: Result after preprocessor finishes processing code file
- What compiler actually compiles
- **Translation**: Entire process of preprocessing, compiling, and linking

### Preprocessor Directives

- **Preprocessor directives** (directives): Instructions starting with `#`, ending with newline (NOT semicolon)
- Tell preprocessor to perform text manipulation tasks
- Have own syntax (sometimes resembles C++, sometimes doesn't)
- Preprocessor doesn't understand C++ syntax
- Final output contains no directives - only processed output passed to compiler

### #include Directive

- Replaces `#include` directive with contents of included file
- Included contents are preprocessed (may recursively include more files)
- Then rest of file is preprocessed

### Macros

#### Overview

- **Macro**: Rule defining how input text converts to replacement output text
- Created with `#define` directive

#### Types of Macros

1. **Object-like macros**
2. **Function-like macros**: Act like functions, generally unsafe, replaceable by normal functions

#### Object-Like Macros - Two Forms

##### With Substitution Text

```cpp
#define IDENTIFIER substitution_text
```

- Used in C to assign names to literals
- No longer necessary (better methods in C++)
- Mostly in legacy code
- **Recommendation**: Avoid whenever possible
- Neither form ends with semicolon (are directives, not statements)

##### Without Substitution Text

```cpp
#define IDENTIFIER
```

- Identifier occurrences replaced by nothing
- Useless for text substitution
- Used for other purposes (conditional compilation)
- Generally acceptable to use

### Conditional Compilation

#### Overview

- Specify conditions for compilation
- Main directives: `#ifdef`, `#ifndef`, `#endif`

#### #ifdef (if defined)

- Checks if identifier previously `#define`d
- If yes: code between `#ifdef` and `#endif` compiled
- If no: code ignored

Example:

```cpp
#define PRINT_JOE

#ifdef PRINT_JOE
    std::cout << "Joe\n"; // compiled (PRINT_JOE defined)
#endif

#ifdef PRINT_BOB
    std::cout << "Bob\n"; // excluded (PRINT_BOB not defined)
#endif
```

#### #ifndef (if not defined)

- Opposite of `#ifdef`
- Checks if identifier NOT `#define`d

#### Alternative Syntax

- `#if defined(IDENTIFIER)` same as `#ifdef IDENTIFIER`
- `#if !defined(IDENTIFIER)` same as `#ifndef IDENTIFIER`
- More C++-style syntax

#### #if 0

- Excludes code block from compilation (like comment block)
- Useful for "commenting out" code containing multi-line comments
- Multi-line comments are non-nestable, so this provides workaround

### Macro Substitution in Preprocessor Commands

- Macro substitution does NOT occur when macro identifier used within another preprocessor command
- Example: `#ifdef PRINT_JOE` doesn't replace `PRINT_JOE` with its substitution

### Scope of #defines

#### File-Based Resolution

- Directives resolved before compilation
- Top to bottom, file by file basis
- Preprocessor doesn't understand C++ concepts (like functions)

Example:

```cpp
void foo()
{
#define MY_NAME "Alex"  // Looks inside function but isn't
}
```

- Behaves as if `#define` was before or after function
- **Best practice**: `#define` identifiers outside functions to avoid confusion

#### Directive Lifetime

- Once preprocessor finishes, all defined identifiers from file are discarded
- Directives only valid from definition point to end of file
- Directives in one file don't impact other files (unless `#include`d)

Example demonstrating separate file scopes:

- `function.cpp` with `#ifdef PRINT` checks its own definitions
- `main.cpp` can `#define PRINT` but won't affect function.cpp's compilation
- Result: function.cpp prints "Not printing!" despite main.cpp defining PRINT

## Header Files

### Purpose & Usage

- `#include` inserts file contents at inclusion point
- Useful way to pull in declarations from another file

### Creating Header Files

- Command line: Create new file in same directory as source files
- **Important**: Don't add to compile command (implicitly included via `#include`, compiled as part of source files)

### Naming Convention

- **Best practice**: Use `.h` suffix (unless project has different convention)
- Longstanding C++ convention
- Most IDEs default to `.h`

### Header-Source Pairing

- Headers often paired with code files
- Header provides forward declarations for corresponding code file
- Example: `add.h` paired with `add.cpp`

**Best practice**: If header paired with code file, use same base name for both

### Including Header Files

#### Syntax

- Use quotes (not angle brackets) for user-defined headers: `#include "add.h"`

#### Source Files Should Include Paired Header

**Best practice**: Source files should `#include` their paired header (if exists)

##### Why?

- Allows compiler to catch errors at compile-time instead of link-time
- Example: If you change function definition in source but forget to update header
    - With include: Compile error (caught early)
    - Without include: Link error (caught late, sees two different definitions)

##### Limitation

- Doesn't work for parameter type mismatches (only return type mismatches)
- C++ supports function overloading - compiler assumes different parameter types = different overload

##### Additional Benefits

- Content required by source file often defined in paired header
- Including header becomes necessity in these cases

### Header File Content Rules

#### Definitions to Avoid

**Best practice**: Do NOT put function and variable definitions in headers (for now)

##### Why?

- Violates One Definition Rule (ODR) when header included in multiple source files
- Will encounter exceptions later (safe to define in headers):
    - Inline functions
    - Inline variables
    - Types
    - Templates

#### Do Not #include .cpp Files

- Preprocessor will do it, but generally shouldn't
- Should add .cpp files to project and compile them

##### Reasons Against Including .cpp

1. Can cause naming collisions between source files
2. Hard to avoid ODR issues in large projects
3. Changes to .cpp cause it AND any file including it to recompile (slow)
4. Headers change less often than source files
5. Non-conventional

### Include Directories

#### Angled Brackets vs Quotes

- **Angled brackets** `<>`: Header file you didn't write
    
    - Preprocessor searches only include directories
    - Include directories: configured in project/IDE/compiler settings
    - Typically default to directories with compiler/OS headers
    - Won't search project's source code directory
- **Double quotes** `""`: Header files you wrote or in current directory
    
    - Searches current directory first
    - Then searches include directories

**Best practice**:

- Double quotes: Headers you wrote or expected in current directory
- Angled brackets: Headers from compiler, OS, or third-party libraries installed elsewhere

### Standard Library Header Evolution

#### Historical Timeline

Original C++ (all in global namespace):

|Header type|Convention|Example|Namespace|
|---|---|---|---|
|C++ specific|`<xxx.h>`|`iostream.h`|Global|
|C compatibility|`<xxx.h>`|`stddef.h`|Global|

#### Standardization Problem

- ANSI committee moved standard library names to `std` namespace
- Would break all old programs using `.h` headers
- Solution: Introduced headerless versions

#### Modern C++ (4 header sets)

|Header type|Convention|Example|Namespace|
|---|---|---|---|
|C++ specific (new)|`<xxx>`|`iostream`|`std`|
|C compatibility (new)|`<cxxx>`|`cstddef`|`std` (required), global (optional)|
|C++ specific (old)|`<xxx.h>`|`iostream.h`|Global|
|C compatibility (old)|`<xxx.h>`|`stddef.h`|Global (required), `std` (optional)|

**Warning**: New C compatibility headers may optionally declare in global namespace; old C compatibility headers may optionally declare in `std` namespace. Avoid these optional locations - may not be declared there on other implementations.

**Best practice**: Use standard library headers without `.h` extension. User-defined headers should still use `.h`.

### Including from Other Directories

#### Bad Method (Relative Paths)

```cpp
#include "headers/myHeader.h"
#include "../moreHeaders/myOtherHeader.h"
```

- Will compile if files exist
- Downside: Reflects directory structure in code
- Updating directory structure breaks code

#### Good Method (Include Paths)

- Tell compiler/IDE about additional header locations
- Set as include path or search directory in project settings
- Compiler searches there when not found in current directory

##### GCC Example

```bash
g++ -o main -I./source/includes main.cpp
```

- No space after `-I`
- Remove `.` after `-I` for full path

##### VS Code Example (tasks.json)

```json
"-I./source/includes",
```

- Add to "Args" section
- No space after `-I`
- Remove `.` for full path

##### Advantage

- Change one compiler/IDE setting instead of every code file

### Header Dependencies

#### When Headers Need Other Headers

- Header file content may use declarations/definitions from another header
- That header should `#include` the other header with needed content

Example:

```cpp
// Foo.h
#include <string_view>  // Required for std::string_view

std::string_view getApplicationName(); // Uses std::string_view
```

### Transitive Includes

- **Transitive includes**: Headers included implicitly (via other headers)
- When source file `#include`s header, also gets headers that header includes
- And so on recursively

**Best practice**: Each file should explicitly `#include` all headers it needs to compile. Don't rely on transitive includes.

### Include Order

#### If Headers Written Properly

- Should `#include` everything they need
- Order shouldn't matter

#### Problem Scenario

- Header A needs declarations from header B but forgets to include
- If code includes B before A: Still compiles! (declarations from B already available)
- Hides the missing include error

#### Recommended Order

**Best practice**: Order `#include`s as follows (skip non-relevant):

1. Paired header for this code file (`add.cpp` includes `"add.h"`)
2. Other headers from same project (`#include "mymath.h"`)
3. Third-party library headers (`#include <boost/tuple/tuple.hpp>`)
4. Standard library headers (`#include <iostream>`)

- Sort alphabetically within each grouping (unless 3rd party docs say otherwise)
- Benefit: Missing `#include` in user headers more likely to cause compile error

## Header Guards

### The Problem

- Header included multiple times causes duplicate definitions
- Example: main.cpp includes both square.h and wave.h, wave.h also includes square.h
- Result: square.h content included twice in main.cpp
- Causes compile error

### Solution: Header Guards (Include Guards)

Conditional compilation directives in this form:

```cpp
#ifndef SOME_UNIQUE_NAME_HERE
#define SOME_UNIQUE_NAME_HERE

// declarations and certain definitions here

#endif
```

#### How It Works

1. First inclusion: `SOME_UNIQUE_NAME_HERE` not defined
    - Gets `#define`d
    - Content included
2. Second inclusion: `SOME_UNIQUE_NAME_HERE` already defined
    - `#ifndef` fails
    - Content ignored

### Header Guard Conventions

- **Best practice**: All header files should have header guards
- Name convention: Full filename in ALL CAPS, underscores for spaces/punctuation
- Example for `square.h`:

```cpp
#ifndef SQUARE_H
#define SQUARE_H

int getSquareSides()
{
    return 4;
}

#endif
```

### Standard Library Example

Visual Studio's iostream:

```cpp
#ifndef _IOSTREAM_
#define _IOSTREAM_

// content

#endif
```

### Advanced: Guard Name Conflicts

#### Problem

- Large programs may have same filename in different directories
- Example: `directoryA\config.h` and `directoryB\config.h`
- Both using `CONFIG_H` guard
- File including both won't receive second config.h content
- Likely causes compilation error

#### Solutions (Complex Guard Names)

- `PROJECT_PATH_FILE_H`
- `FILE_LARGE-RANDOM-NUMBER_H`
- `FILE_CREATION-DATE_H`

### What Header Guards Don't Prevent

- Don't prevent header from being included once into different code files
- By design - this is allowed
- Can still cause unexpected problems when definitions spread across files

### Why Header Guards Matter

- Even though we avoid function definitions in headers
- Still need guards for future cases:
    - Custom type definitions (must be in headers)
    - Without guards: multiple identical type definition copies
    - Compiler flags as error
- Establishing good habits now

## #pragma once

### Overview

```cpp
#pragma once

// your code here
```

- Modern alternative to traditional header guards
- Same purpose: avoid multiple inclusion
- Request compiler to guard header (implementation-specific how)

### Comparison to Header Guards

- Traditional: Developer responsible for guarding (using `#ifndef`, `#define`, `#endif`)
- `#pragma once`: Compiler handles it

### Known Failure Case

- Header copied to exist in multiple file system locations
- Both copies somehow get included
- Header guards: Successfully de-dupe (identical content)
- `#pragma once`: Fails (doesn't realize identical content)
    - Probably generates unique ID per file
    - Different files get different IDs
    - Compiler treats as different headers

### Advantages

- Easier to use
- Less error-prone
- Many developers prefer it
- Many IDEs auto-include at top of new headers

### Portability Concern

**Warning**: `#pragma` directive designed for compiler implementers' purposes. Support and meaning completely implementation-specific. Except `#pragma once`, don't expect pragmas to work across compilers.

- Not in C++ standard
- Some compilers may not implement it
- Some organizations (like Google) recommend traditional header guards
- Support fairly ubiquitous now
- Generally accepted in modern C++

### Tutorial Approach

- Will favor traditional header guards (most conventional)
- Using `#pragma once` instead is acceptable in modern C++