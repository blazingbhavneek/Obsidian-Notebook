
## 1. Language Levels & Execution

### Assembly Language

- **Human-readable format** of machine language
- **Architecture-specific**: Each CPU architecture (x86, ARM, MIPS, PowerPC) has its own assembly language
- **Very low-level**: Requires deep knowledge of CPU architecture
- **No portability**: Code written for one CPU family cannot run on another without complete rewrite
- **Direct hardware control**: Provides precise control over CPU registers and instructions

### Compiled vs Interpreted Languages

**Compiled Languages (e.g., C++, Rust, Go)**

- Source code → Compiler → Machine code (executable)
- Fast execution (no runtime translation needed)
- Requires recompilation for changes
- Platform-specific executables

**Interpreted Languages (e.g., Python, JavaScript)**

- Source code → Interpreter → Executes line-by-line at runtime
- **Flexible**: No compilation step needed; edit and run immediately
- **Slower**: Interpretation overhead at runtime
- **Requires interpreter**: Must be installed on target device
- Platform-independent source code

**Note**: C++ can be compiled to assembly code first (useful for debugging/optimization), but typically compiles directly to machine code.

## 2. Cross-Platform Compilation

### How C++ Achieves Portability

- **High-level abstractions**: C++ code is hardware-agnostic
- **Multiple compilers**: Different compilers exist for different platforms:
    - **x86/x64**: GCC, Clang, MSVC, Intel C++
    - **ARM**: ARM Compiler, GCC ARM, Clang
    - **PowerPC**: GCC PowerPC
    - **MIPS**: GCC MIPS
- **Same source, different executables**: Write once, compile for each target platform
- **Hardware support required**: Target platform must support the program's requirements

## 3. Language Standards

### The C Standard

- **ANSI C** (1989): First standardized version
- **ISO C90** (1990): International standard, identical to ANSI C
- **C99, C11, C17, C23**: Later revisions

**Standards Compliance**: When a compiler is "compliant" with a standard, it means:

- Implements all required language features correctly
- Follows specified behavior for all constructs
- Produces expected results for standard-conforming code

### The C++ Standard

- **Evolution**: C++98 → C++03 → C++11 → C++14 → C++17 → C++20 → C++23 → C++26 (upcoming)
- **ISO Standard**: International Organization for Standardization maintains it
- **Major versions**:
    - **C++11**: Modern C++ begins (auto, lambdas, smart pointers, move semantics)
    - **C++14**: Bug fixes and minor improvements
    - **C++17**: Structured bindings, filesystem, parallel algorithms
    - **C++20**: Concepts, ranges, coroutines, modules
    - **C++23**: Latest finalized standard

### Understanding C++ Versions

**What does "choosing a standard" mean?**

- Tells the compiler which language features to recognize
- C++23 code uses features **not available** in C++17
- Cannot treat C++23 code with C++17 standard (compiler will error on unknown features)
- **Can** compile C++17 code with C++23 standard (backward compatible)

**Rule of thumb**: Newer standards are backward compatible, but older compilers can't understand newer features.

**Professional Best Practice**:

- Use a standard **1-2 versions behind** the latest (e.g., C++17/C++20 when C++23 is latest)
- Allows time for compiler bugs to be fixed
- Best practices for new features become established
- Better tooling support

## 4. The Compilation Process

### Step 1: Compilation

The **C++ compiler** processes each source file (`.cpp`) individually:

1. **Preprocessing**: Handles `#include`, `#define`, conditional compilation
2. **Syntax checking**: Verifies C++ language rules are followed
3. **Translation**: Converts C++ code to machine code
4. **Output**: Produces object file (`.o` or `.obj`)

### Step 2: Linking

The **linker** combines everything:

1. **Resolves dependencies**: Connects function calls across different files
2. **Links libraries**: Includes pre-compiled code (standard library, third-party libs)
3. **Creates executable**: Produces final runnable program

### Step 3: Building

- **Building**: The complete process (compile + link)
- **Build**: The resulting executable
- **Build automation tools**: `make`, `CMake`, `build2`, `Ninja`

### GCC vs G++

|Tool|Purpose|
|---|---|
|`gcc`|C compiler (can compile C++ syntax, but **won't link C++ libs**)|
|`g++`|C++ compiler + linker (automatically adds `-lstdc++`)|

**Example**:

```bash
# Compile and link in one step
g++ main.cpp -o main

# Run the executable
./main
# Output: Hello World
```

## 5. Build Configurations

### Debug Configuration

- **Purpose**: Development and debugging
- **Settings**:
    - **Optimizations**: OFF (`-O0`)
    - **Debug info**: ON (`-g` or `-ggdb`)
    - **Assertions**: Enabled
- **Result**: Larger, slower executable that's easier to debug

### Release Configuration

- **Purpose**: Production deployment
- **Settings**:
    - **Optimizations**: ON (`-O2` or `-O3`)
    - **Debug info**: OFF
    - **Assertions**: Disabled (`-DNDEBUG`)
- **Result**: Smaller, faster executable

### VS Code Configuration

**Debug build** (`tasks.json`, add to `args` before `"${file}"`):

```json
"-ggdb",
```

**Release build**:

```json
"-O2",
"-DNDEBUG",
```

## 6. Compiler Extensions & Standards Compliance

### Compiler Extensions

- **What**: Non-standard features added by compiler vendors
- **Examples**:
    - GNU extensions in GCC (`__attribute__`, statement expressions)
    - MSVC-specific keywords (`__declspec`)
- **Problem**: Makes code non-portable
- **Solution**: Use `-pedantic-errors` flag

### Enforcing Standards Compliance

**VS Code** (`tasks.json`):

```json
"-pedantic-errors",
```

This makes the compiler reject non-standard code.

## 7. Compiler Diagnostics

### Diagnostic Errors (Compilation Errors)

- **Meaning**: Compiler stops; cannot proceed
- **Examples**:
    - Syntax errors
    - Type mismatches
    - Missing semicolons
- **Result**: No executable produced
- **Also called**: Compile errors, compiler errors

### Diagnostic Warnings

- **Meaning**: Compiler finds issues but continues
- **Examples**:
    - Unused variables
    - Implicit conversions
    - Deprecated features
- **Result**: Executable produced, but may behave unexpectedly

### Warning Suppression

- C++ has **no official way** to suppress warnings
- Compilers provide **non-portable solutions**:
    - GCC/Clang: `#pragma GCC diagnostic ignored`
    - MSVC: `#pragma warning(disable: xxxx)`

### Recommended Warning Flags (VS Code)

Add to `tasks.json` `args`:

```json
"-Wall",           // Enable most warnings
"-Weffc++",        // Effective C++ guidelines
"-Wextra",         // Extra warnings
"-Wconversion",    // Implicit conversions
"-Wsign-conversion", // Sign conversions
"-Werror"          // Treat warnings as errors
```

**Best Practice**: Enable "Treat warnings as errors" (`-Werror`) to force resolution of all issues.

## 8. Setting Language Standard

### Default Behavior

- Most compilers default to **C++14** or older
- Missing modern features (C++17/20/23)

### GCC/G++/Clang Flags

```bash
-std=c++11   # C++11
-std=c++14   # C++14
-std=c++17   # C++17
-std=c++20   # C++20 (use -std=c++2a for GCC 8/9)
-std=c++23   # C++23
-std=c++2c   # Experimental C++26 features
```

### VS Code Configuration

1. **Compiler** (`tasks.json`, add to `args`):

```json
"-std=c++20",
```

2. **IntelliSense** (`settings.json`):

```json
"C_Cpp.default.cppStandard": "c++20"
```

### Checking Your Standard

Use this program to verify which standard your compiler uses:

```cpp
#include <iostream>

const int numStandards = 7;
const long stdCode[numStandards] = { 199711L, 201103L, 201402L, 201703L, 202002L, 202302L, 202612L};
const char* stdName[numStandards] = { "Pre-C++11", "C++11", "C++14", "C++17", "C++20", "C++23", "C++26" };

long getCPPStandard()
{
#if defined (_MSVC_LANG)
    return _MSVC_LANG;  // Visual Studio
#elif defined (_MSC_VER)
    return -1;  // Older Visual Studio
#else
    return __cplusplus;  // Standard way
#endif
}

int main()
{
    long standard = getCPPStandard();

    if (standard == -1)
    {
        std::cout << "Error: Unable to determine language standard.\n";
        return 0;
    }

    for (int i = 0; i < numStandards; ++i)
    {
        if (standard == stdCode[i])
        {
            std::cout << "Using " << stdName[i]
                << " (code " << standard << "L)\n";
            break;
        }

        if (standard < stdCode[i])
        {
            std::cout << "Using preview of " << stdName[i]
                << " (code " << standard << "L)\n";
            break;
        }
    }

    return 0;
}
```

## 9. Useful Code Snippets

### Wait for Enter Key (Console Programs)

```cpp
std::cin.clear(); // Reset error flags
std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear buffer
std::cin.get(); // Wait for Enter
```

**Note**: Requires `#include <limits>` for `std::numeric_limits`.

## 10. Quick Reference: Optimal VS Code Setup

**Complete `tasks.json` args section**:

```json
"args": [
    "-std=c++20",              // Use C++20 standard
    "-pedantic-errors",        // Strict standards compliance
    "-Wall",                   // Most warnings
    "-Weffc++",                // Effective C++ warnings
    "-Wextra",                 // Extra warnings
    "-Wconversion",            // Conversion warnings
    "-Wsign-conversion",       // Sign conversion warnings
    "-Werror",                 // Treat warnings as errors
    "-ggdb",                   // Debug info (remove for release)
    "-g",                      // General debug info
    "${file}",                 // Input file
    "-o",                      // Output flag
    "${fileDirname}/${fileBasenameNoExtension}"  // Output file
],
```

**For release builds**, replace `-ggdb` and `-g` with:

```json
"-O2",        // Optimization level 2
"-DNDEBUG",   // Disable assertions
```

---

## Key Takeaways

1. **C++ is compiled**, not interpreted - fast execution but platform-specific
2. **Standards matter** - use modern standards (C++17/20) for better features
3. **Enable warnings** - catch issues early with `-Wall -Wextra -Werror`
4. **Use build configurations** - debug during development, release for production
5. **Standards compliance** - use `-pedantic-errors` for portable code
6. **Backward compatibility** - newer standards support older code, not vice versa