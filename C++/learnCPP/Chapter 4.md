
##  fundamental data types

variables are names for a piece of memory that can be used to store information

computers have random access memory (RAM) that is available for programs to use. When a variable is defined, a piece of that memory is set aside for that variable

Memory is organized into sequential units called **memory addresses** (or **addresses** for short
Instead, each memory address holds 1 byte of data. A **byte** is a group of bits that are operated on as a unit

As an aside…

Some older or non-standard machines may have bytes of a different size (from 1 to 48 bits) -- however, we generally need not worry about these, as the modern de-facto standard is that a byte is 8 bits. For these tutorials, we’ll assume a byte is 8 bits.

Because all data on a computer is just a sequence of bits, we use a **data type** (often called a **type** for short) to tell the compiler how to interpret the contents of memory in some meaningful way

The C++ language comes with many predefined data types available for your use. The most basic of these types are called the **fundamental data types** (informally sometimes called **basic types** or **primitive types**).

|Types|Category|Meaning|Example|
|---|---|---|---|
|float  <br>double  <br>long double|Floating Point|a number with a fractional part|3.14159|
|bool|Integral (Boolean)|true or false|true|
|char  <br>wchar_t  <br>char8_t (C++20)  <br>char16_t (C++11)  <br>char32_t (C++11)|Integral (Character)|a single character of text|‘c’|
|short int  <br>int  <br>long int  <br>long long int (C++11)|Integral (Integer)|positive and negative whole numbers, including 0|64|
|std::nullptr_t (C++11)|Null Pointer|a null pointer|nullptr|
|void|Void|no type|n/a|

Integer vs integral types

In mathematics, an “integer” is a number with no decimal or fractional part, including negative and positive numbers and zero. The term “integral” has several different meanings, but in the context of C++ is used to mean “like an integer”.

The C++ standard defines the following terms:

- The **standard integer types** are `short`, `int`, `long`, `long long` (including their signed and unsigned variants).
- The **integral types** are `bool`, the various char types, and the standard integer types.

All integral types are stored in memory as integer values, but only the standard integer types will display as an integer value when output. We’ll discuss what `bool` and the char types do when output in their respective lessons.

The C++ standard also explicitly notes that “integer types” is a synonym for “integral types”. However, conventionally, “integer types” is more often used as shorthand for the “standard integer types” instead.

Also note that the term “integral types” only includes fundamental types. This means non-fundamental types (such as `enum` and `enum class`) are not integral types, even when they are stored as an integer (and in the case of `enum`, displayed as one too).


C++ contains three sets of types.

The first two are built-in to the language itself (and do not require the inclusion of a header to use):

- The “fundamental data types” provide the most the basic and essential data types.
- The “compound data types” provide more complex data types and allow for the creation of custom (user-defined) types. We cover these in lesson [12.1 -- Introduction to compound data types](https://www.learncpp.com/cpp-tutorial/introduction-to-compound-data-types/).

The distinction between the fundamental and compound types isn’t all that interesting or relevant, so it’s generally fine to consider them as a single set of types.

The third (and largest) set of types is provided by the C++ standard library. Because the standard library is included in all C++ distributions, these types are broadly available and have been standardized for compatibility. Use of the types in the standard library requires the inclusion of the appropriate header and linking in the standard library.


Nomenclature

The term “built-in type” is most often used as a synonym for the fundamental data types. However, Stroustrup and others use the term to mean both the fundamental and compound data types (both of which are built-in to the core language). Since this term isn’t well-defined, we recommend avoiding it accordingly.

The _t suffix

Many of the types defined in newer versions of C++ (e.g. `std::nullptr_t`) use a _t suffix. This suffix means “type”, and it’s a common nomenclature applied to modern types.

If you see something with a _t suffix, it’s probably a type. But many types don’t have a _t suffix, so this isn’t consistently applied.



Void is our first example of an incomplete type. An **incomplete type** is a type that has been declared but not yet defined

`void` is intentionally incomplete since it represents the lack of a type, and thus cannot be defined


Deprecated: Functions that do not take parameters

In C, void is used as a way to indicate that a function does not take any parameters:

```cpp
int getValue(void) // void here means no parameters
{
    int x{};
    std::cin >> x;

    return x;
}
```

Although this will compile in C++ (for backwards compatibility reasons), this use of keyword _void_ is considered deprecated in C++. The following code is equivalent, and preferred in C++:

```cpp
int getValue() // empty function parameters is an implicit void
{
    int x{};
    std::cin >> x;

    return x;
}
```

most objects actually take up more than 1 byte of memory. A single object may use 1, 2, 4, 8, or even more consecutive memory addresses. The amount of memory that an object uses is based on its data type.

Because we typically access memory through variable names (and not directly via memory addresses), the compiler is able to hide the details of how many bytes a given object uses from us. When we access some variable `x` in our source code, the compiler knows how many bytes of data need to be retrieved (based on the type of variable `x`), and will output the appropriate machine language code to handle that detail for us.


New programmers often focus too much on optimizing their code to use as little memory as possible. In most cases, this makes a negligible difference. Focus on writing maintainable code, and optimize only when and where the benefit will be substantive.


Fundamental data type sizes

The obvious next question is “how much memory do objects of a given data type use?”. Perhaps surprisingly, the C++ standard does not define the exact size (in bits) of any of the fundamental types.

Instead, the standard says the following:

- An object must occupy at least 1 byte (so that each object has a distinct memory address).
- A byte must be at least 8 bits.
- The integral types `char`, `short`, `int`, `long`, and `long long` have a minimum size of 8, 16, 16, 32, and 64 bits respectively.
- `char` and `char8_t` are exactly 1 byte (at least 8 bits).


Nomenclature

When we talk about the size of a type, we really mean the size of an instantiated object of that type.

- A byte is 8 bits.
- Memory is byte addressable (we can access every byte of memory independently).
- Floating point support is IEEE-754 compliant.
- We are on a 32-bit or 64-bit architecture.

Tip

For maximum portability, you shouldn’t assume that objects are larger than the specified minimum size.

Alternatively, if you want to assume that a type has some non-minimum size (e.g. that an int is at least `4` bytes), you can use `static_assert` to have the compiler fail a build if it is compiled on an architecture where this assumption is not true. We cover how to do this in lesson [9.6 -- Assert and static_assert](https://www.learncpp.com/cpp-tutorial/assert-and-static_assert/#static_assert).


The sizeof operator

In order to determine the size of data types on a particular machine, C++ provides an operator named `sizeof`. The **sizeof operator** is a unary operator that takes either a type or a variable, and returns the size of an object of that type (in bytes). You can compile and run the following program to find out how large some of your data types are:

```cpp
#include <iomanip> // for std::setw (which sets the width of the subsequent output)
#include <iostream>
#include <climits> // for CHAR_BIT

int main()
{
    std::cout << "A byte is " << CHAR_BIT << " bits\n\n";

    std::cout << std::left; // left justify output

    std::cout << std::setw(16) << "bool:" << sizeof(bool) << " bytes\n";
    std::cout << std::setw(16) << "char:" << sizeof(char) << " bytes\n";
    std::cout << std::setw(16) << "short:" << sizeof(short) << " bytes\n";
    std::cout << std::setw(16) << "int:" << sizeof(int) << " bytes\n";
    std::cout << std::setw(16) << "long:" << sizeof(long) << " bytes\n";
    std::cout << std::setw(16) << "long long:" << sizeof(long long) << " bytes\n";
    std::cout << std::setw(16) << "float:" << sizeof(float) << " bytes\n";
    std::cout << std::setw(16) << "double:" << sizeof(double) << " bytes\n";
    std::cout << std::setw(16) << "long double:" << sizeof(long double) << " bytes\n";

    return 0;
}
```

For gcc users

If you have not disabled compiler extensions, gcc allows `sizeof(void)` to return 1 instead of producing a diagnostic ([Pointer-Arith](https://gcc.gnu.org/onlinedocs/gcc-4.4.2/gcc/Pointer-Arith.html#Pointer-Arith)). We show how to disable compiler extensions in lesson [0.10 -- Configuring your compiler: Compiler extensions](https://www.learncpp.com/cpp-tutorial/configuring-your-compiler-compiler-extensions/).


For advanced readers

`sizeof` does not include dynamically allocated memory used by an object. We discuss dynamic memory allocation in a future lesson.


As an aside…

You might assume that types that use less memory would be faster than types that use more memory. This is not always true. CPUs are often optimized to process data of a certain size (e.g. 32 bits), and types that match that size may be processed quicker. On such a machine, a 32-bit int could be faster than a 16-bit short or an 8-bit char.

An **integer** is an integral type that can represent positive and negative whole numbers, including 0 (e.g. -2, -1, 0, 1, 2). C++ has _4_ primary fundamental integer types available for use:


|Type|Minimum Size|Note|
|---|---|---|
|short int|16 bits||
|int|16 bits|Typically 32 bits on modern architectures|
|long int|32 bits||
|long long int|64 bits||

A reminder

C++ only guarantees that integers will have a certain minimum size, not that they will have a specific size. See lesson [4.3 -- Object sizes and the sizeof operator](https://www.learncpp.com/cpp-tutorial/object-sizes-and-the-sizeof-operator/) for information on how to determine how large each type is on your machine.

As an aside…

Technically, the `bool` and `char` types are considered to be integral types (because these types store their values as integer values). For the purpose of the next few lessons, we’ll exclude these types from our discussion.

Defining signed integers

Here is the preferred way to define the four types of signed integers:

```cpp
short s;      // prefer "short" instead of "short int"
int i;
long l;       // prefer "long" instead of "long int"
long long ll; // prefer "long long" instead of "long long int"
```

The integer types can also take an optional _signed_ keyword, which by convention is typically placed before the type name:

```cpp
signed short ss;
signed int si;
signed long sl;
signed long long sll;
```

However, this keyword should not be used, as it is redundant, since integers are signed by default.

|Size / Type|Range|
|---|---|
|8-bit signed|-128 to 127|
|16-bit signed|-32,768 to 32,767|
|32-bit signed|-2,147,483,648 to 2,147,483,647|
|64-bit signed|-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807|

For advanced readers

The above ranges assume “two’s complement” binary representation. This representation is the de-facto standard for modern architectures (as it is easier to implement in hardware), and is now required by the C++20 standard. We discuss two’s complement in lesson [O.4 -- Converting integers between binary and decimal representation](https://www.learncpp.com/cpp-tutorial/converting-integers-between-binary-and-decimal-representation/).

In prior standards, sign-magnitude and ones complement representations were permitted for historical reasons. Such representations produce values in the range -(2n-1-1) to +(2n-1-1).

Twos complement and ones complement representation difference?

Overflow

What happens if we try to assign the value _140_ to an 8-bit signed integer? This number is outside the range that an 8-bit signed integer can hold. The number 140 requires 9 bits to represent (8 magnitude bits and 1 sign bit), but we only have 8 bits (7 magnitude bits and 1 sign bit) available in an 8-bit signed integer.

The C++20 standard makes this blanket statement: “If during the evaluation of an expression, the result is not mathematically defined or not in the range of representable values for its type, the behavior is undefined”. Colloquially, this is called **overflow**.

Therefore, assigning value 140 to an 8-bit signed integer will result in undefined behavior.

If an arithmetic operation (such as addition or multiplication) attempts to create a value outside the range that can be represented, this is called **integer overflow** (or **arithmetic overflow**). For signed integers, integer overflow will result in undefined behavior.


For advanced readers

We cover what happens when unsigned integers overflow in lesson [4.5 -- Unsigned integers, and why to avoid them](https://www.learncpp.com/cpp-tutorial/unsigned-integers-and-why-to-avoid-them/).


## Unsigned integers, and why to avoid them

Defining unsigned integers

To define an unsigned integer, we use the _unsigned_ keyword. By convention, this is placed before the type:

```cpp
unsigned short us;
unsigned int ui;
unsigned long ul;
unsigned long long ull;
```


When no negative numbers are required, unsigned integers are well-suited for networking and systems with little memory, because unsigned integers can store more positive numbers without taking up extra memory.

Author’s note

Oddly, the C++ standard explicitly says “a computation involving unsigned operands can never overflow”. This is contrary to general programming consensus that integer overflow encompasses both signed and unsigned use cases ([cite](https://en.wikipedia.org/wiki/Integer_overflow#Definition_variations_and_ambiguity)). Given that most programmers would consider this overflow, we’ll call this overflow despite the C++ standard’s statements to the contrary.

If an unsigned value is out of range, it is divided by one greater than the largest number of the type, and only the remainder kept.

The number `280` is too big to fit in our 1-byte range of 0 to 255. 1 greater than the largest number of the type is 256. Therefore, we divide 280 by 256, getting 1 remainder 24. The remainder of 24 is what is stored.

Here’s another way to think about the same thing. Any number bigger than the largest number representable by the type simply “wraps around” (sometimes called “modulo wrapping”). `255` is in range of a 1-byte integer, so `255` is fine. `256`, however, is outside the range, so it wraps around to the value `0`. `257` wraps around to the value `1`. `280` wraps around to the value `24`.


The controversy over unsigned numbers

Many developers (and some large development houses, such as Google) believe that developers should generally avoid unsigned integers.

This is largely because of two behaviors that can cause problems.

First, with signed values, it takes a little work to accidentally overflow the top or bottom of the range because those values are far from 0. With unsigned numbers, it is much easier to overflow the bottom of the range, because the bottom of the range is 0, which is close to where the majority of our values are.


Second, and more insidiously, unexpected behavior can result when you mix signed and unsigned integers. In C++, if a mathematical operation (e.g. arithmetic or comparison) has one signed integer and one unsigned integer, the signed integer will usually be converted to an unsigned integer. And the result will thus be unsigned


The author of doSomething() was expecting someone to call this function with only positive numbers. But the caller is passing in _-1_ -- clearly a mistake, but one made regardless. What happens in this case?

The signed argument of `-1` gets implicitly converted to an unsigned parameter. `-1` isn’t in the range of an unsigned number, so it wraps around to 4294967295. Then your program goes ballistic.

Even more problematically, it can be hard to prevent this from happening. Unless you’ve configured your compiler to be aggressive about producing signed/unsigned conversion warnings (and you should), your compiler probably won’t even complain about this.

All of these problems are commonly encountered, produce unexpected behavior, and are hard to find, even using automated tools designed to detect problem cases.

Given the above, the somewhat controversial best practice that we’ll advocate for is to avoid unsigned types except in specific circumstances.

Best practice

Favor signed numbers over unsigned numbers for holding quantities (even quantities that should be non-negative) and mathematical operations. Avoid mixing signed and unsigned numbers.

So when should you use unsigned numbers?

There are still a few cases in C++ where it’s okay / necessary to use unsigned numbers.

First, unsigned numbers are preferred when dealing with bit manipulation (covered in chapter O -- that’s a capital ‘o’, not a ‘0’). They are also useful when well-defined wrap-around behavior is required (useful in some algorithms like encryption and random number generation).

Second, use of unsigned numbers is still unavoidable in some cases, mainly those having to do with array indexing. We’ll talk more about this in the lessons on arrays and array indexing.

Also note that if you’re developing for an embedded system (e.g. an Arduino) or some other processor/memory limited context, use of unsigned numbers is more common and accepted (and in some cases, unavoidable) for performance reasons.


Why isn’t the size of the integer types fixed?

The short answer is that this goes back to the early days of C, when computers were slow and performance was of the utmost concern. C opted to intentionally leave the size of an integer open so that the compiler implementers could pick a size for `int` that performs best on the target computer architecture. That way, programmers could just use `int` without having to worry about whether they could be using something more performant.

By modern standards, the lack of consistent ranges for the various integral types sucks (especially in a language designed to be portable).

Fixed-width integers

To address the above issues, C++11 provides an alternate set of integer types that are guaranteed to be the same size on any architecture. Because the size of these integers is fixed, they are called **fixed-width integers**.

The fixed-width integers are defined (in the \<cstdint>\ header) as follows:

|Name|Fixed Size|Fixed Range|Notes|
|---|---|---|---|
|std::int8_t|1 byte signed|-128 to 127|Treated like a signed char on many systems. See note below.|
|std::uint8_t|1 byte unsigned|0 to 255|Treated like an unsigned char on many systems. See note below.|
|std::int16_t|2 byte signed|-32,768 to 32,767||
|std::uint16_t|2 byte unsigned|0 to 65,535||
|std::int32_t|4 byte signed|-2,147,483,648 to 2,147,483,647||
|std::uint32_t|4 byte unsigned|0 to 4,294,967,295||
|std::int64_t|8 byte signed|-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807||
|std::uint64_t|8 byte unsigned|0 to 18,446,744,073,709,551,615||

Here’s an example:

```cpp
#include <cstdint> // for fixed-width integers
#include <iostream>

int main()
{
    std::int32_t x { 32767 }; // x is always a 32-bit integer
    x = x + 1;                // so 32768 will always fit
    std::cout << x << '\n';

    return 0;
}
```

Best practice

Use a fixed-width integer type when you need an integral type that has a guaranteed range.


Warning: `std::int8_t` and `std::uint8_t` typically behave like chars

Due to an oversight in the C++ specification, modern compilers typically treat `std::int8_t` and `std::uint8_t` (and the corresponding fast and least fixed-width types, which we’ll introduce in a moment) the same as `signed char` and `unsigned char` respectively. Thus on most modern systems, the 8-bit fixed-width integral types will behave like char types.

As a quick teaser:

```cpp
#include <cstdint> // for fixed-width integers
#include <iostream>

int main()
{
    std::int8_t x { 65 };   // initialize 8-bit integral type with value 65
    std::cout << x << '\n'; // You're probably expecting this to print 65

    return 0;
}
```


Although you’re probably expecting the above program to print `65`, it most likely won’t.


Warning

The 8-bit fixed-width integer types are often treated like chars instead of integer values (and this may vary per system). The 16-bit and wider integral types are not subject to this issue.

For advanced readers

The fixed-width integers actually don’t define new types -- they are just aliases for existing integral types with the desired size. For each fixed-width type, the implementation (the compiler and standard library) gets to determine which existing type is aliased. As an example, on a platform where `int` is 32-bits, `std::int32_t` will be an alias for `int`. On a system where `int` is 16-bits (and `long` is 32-bits), `std::int32_t` will be an alias for `long` instead.

So what about the 8-bit fixed-width types?

In most cases, `std::int8_t` is an alias for `signed char` because it is the only available 8-bit signed integral type (`bool` and `char` are not considered to be signed integral types). And when this is the case, `std::int8_t` will behave just like a char on that platform.

However, in rare cases, if a platform has an implementation-specific 8-bit signed integral type, the implementation may decide to make `std::int8_t` an alias for that type instead. In that case, `std::int8_t` will behave like that type, which may be more like an int than a char.

`std::uint8_t` behaves similarly.


Other fixed-width downsides

The fixed-width integers have some potential downsides:

First, the fixed-width integers are not guaranteed to be defined on all architectures. They only exist on systems where there are fundamental integral types that match their widths and following a certain binary representation. Your program will fail to compile on any such architecture that does not support a fixed-width integer that your program is using. However, given that modern architectures have standardized around 8/16/32/64-bit variables, this is unlikely to be a problem unless your program needs to be portable to some exotic mainframe or embedded architectures.

Second, if you use a fixed-width integer, it may be slower than a wider type on some architectures. For example, if you need an integer that is guaranteed to be 32-bits, you might decide to use `std::int32_t`, but your CPU might actually be faster at processing 64-bit integers. However, just because your CPU can process a given type faster doesn’t mean your program will be faster overall -- modern programs are often constrained by memory usage rather than CPU, and the larger memory footprint may slow your program more than the faster CPU processing accelerates it. It’s hard to know without actually measuring.


Fast and least integral types Optional

To help address the above downsides, C++ also defines two alternative sets of integers that are guaranteed to exist.

The fast types (std::int_fast#_t and std::uint_fast#_t) provide the fastest signed/unsigned integer type with a width of at least # bits (where # = 8, 16, 32, or 64). For example, `std::int_fast32_t` will give you the fastest signed integer type that’s at least 32-bits. By fastest, we mean the integral type that can be processed most quickly by the CPU.

The least types (std::int_least#_t and std::uint_least#_t) provide the smallest signed/unsigned integer type with a width of at least # bits (where # = 8, 16, 32, or 64). For example, `std::uint_least32_t` will give you the smallest unsigned integer type that’s at least 32-bits.

Here’s an example from the author’s Visual Studio (32-bit console application):

```cpp
#include <cstdint> // for fast and least types
#include <iostream>

int main()
{
	std::cout << "least 8:  " << sizeof(std::int_least8_t)  * 8 << " bits\n";
	std::cout << "least 16: " << sizeof(std::int_least16_t) * 8 << " bits\n";
	std::cout << "least 32: " << sizeof(std::int_least32_t) * 8 << " bits\n";
	std::cout << '\n';
	std::cout << "fast 8:  "  << sizeof(std::int_fast8_t)   * 8 << " bits\n";
	std::cout << "fast 16: "  << sizeof(std::int_fast16_t)  * 8 << " bits\n";
	std::cout << "fast 32: "  << sizeof(std::int_fast32_t)  * 8 << " bits\n";

	return 0;
}
```

This produced the result:

least 8:  8 bits
least 16: 16 bits
least 32: 32 bits

fast 8:  8 bits
fast 16: 32 bits
fast 32: 32 bits

You can see that `std::int_least16_t` is 16-bits, whereas `std::int_fast16_t` is actually 32-bits. This is because on the author’s machine, 32-bit integers are faster to process than 16-bit integers.

As another example, let’s assume we’re on an architecture that has only 16-bit and 64-bit integral types. `std::int32_t` would not exist, whereas `std::least_int32_t` (and `std::fast_int32_t`) would be 64 bits.

However, these fast and least integers have their own downsides. First, not many programmers actually use them, and a lack of familiarity can lead to errors. Then the fast types can also lead to memory wastage, as their actual size may be significantly larger than indicated by their name.

Most seriously, because the size of the fast/least integers is implementation-defined, your program may exhibit different behaviors on architectures where they resolve to different sizes.


Best practice

Avoid the fast and least integral types because they may exhibit different behaviors on architectures where they resolve to different sizes.


Best practice

- Prefer `int` when the size of the integer doesn’t matter (e.g. the number will always fit within the range of a 2-byte signed integer). For example, if you’re asking the user to enter their age, or counting from 1 to 10, it doesn’t matter whether `int` is 16-bits or 32-bits (the numbers will fit either way). This will cover the vast majority of the cases you’re likely to run across.
- Prefer `std::int#_t` when storing a quantity that needs a guaranteed range.
- Prefer `std::uint#_t` when doing bit manipulation or well-defined wrap-around behavior is required (e.g. for cryptography or random number generation).

Avoid the following when possible:

- `short` and `long` integers (prefer a fixed-width integer type instead).
- The fast and least integral types (prefer a fixed-width integer type instead).
- Unsigned types for holding quantities (prefer a signed integer type instead).
- The 8-bit fixed-width integer types (prefer a 16-bit fixed-width integer type instead).
- Any compiler-specific fixed-width integers (for example, Visual Studio defines __int8, __int16, etc…)

Pretty simple, right? We can infer that operator `sizeof` returns an integer value -- but what integral type is that return value? An int? A short? The answer is that `sizeof` returns a value of type `std::size_t`. **std::size_t** is an alias for an implementation-defined unsigned integral type. In other words, the compiler decides if `std::size_t` is an unsigned int, an unsigned long, an unsigned long long, etc…

Key insight

`std::size_t` is an alias for an implementation-defined unsigned integral type. It is used within the standard library to represent the byte-size or length of objects.

For advanced readers

`std::size_t` is actually a typedef. We cover typedefs in lesson [10.7 -- Typedefs and type aliases](https://www.learncpp.com/cpp-tutorial/typedefs-and-type-aliases/).


```cpp
#include <cstddef>  // for std::size_t
#include <iostream>

int main()
{
    int x { 5 };
    std::size_t s { sizeof(x) }; // sizeof returns a value of type std::size_t, so that should be the type of s
    std::cout << s << '\n';

    return 0;
}
```

Best practice

If you use `std::size_t` explicitly in your code, #include one of the headers that defines `std::size_t` (we recommend \<cstddef>).

Using `sizeof` does not require a header (even though it returns a value whose type is `std::size_t`).

Much like an integer can vary in size depending on the system, `std::size_t` also varies in size. `std::size_t` is guaranteed to be unsigned and at least 16 bits, but on most systems will be equivalent to the address-width of the application. That is, for 32-bit applications, `std::size_t` will typically be a 32-bit unsigned integer, and for a 64-bit application, `std::size_t` will typically be a 64-bit unsigned integer.


The `sizeof` operator returns a value of type `std::size_t` Optional

Author’s note

The following sections are optional reading. It is not critical that you understand what follows.

Amusingly, we can use the `sizeof` operator (which returns a value of type `std::size_t`) to ask for the size of `std::size_t` itself:

```cpp
#include <cstddef> // for std::size_t
#include <iostream>

int main()
{
	std::cout << sizeof(std::size_t) << '\n';

	return 0;
}
```

Compiled as a 32-bit (4 byte) console app on the author’s system, this prints:

4

`std::size_t` imposes an upper limit on the size of an object Optional

The `sizeof` operator must be able to return the byte-size of an object as a value of type `std::size_t`. Therefore, the byte-size of an object can be no larger than the largest value `std::size_t` can hold.

The [C++20 standard](https://isocpp.org/files/papers/N4860.pdf#subsection.6.8.2) ([basic.compound] 1.8.2) says: “Constructing a type such that the number of bytes in its object representation exceeds the maximum value representable in the type std::size_t (17.2) is ill-formed.”

If it were possible to create a larger object, `sizeof` would not be able to return its byte-size, as it would be outside the range that a `std::size_t` could hold. Thus, creating an object with a size (in bytes) larger than the largest value an object of type `std::size_t` can hold is invalid (and will cause a compile error).

For example, let’s assume that `std::size_t` has a size of 4 bytes on our system. An unsigned 4-byte integral type has range 0 to 4,294,967,295. Therefore, a 4-byte `std::size_t` object can hold any value from 0 to 4,294,967,295. Any object with a byte-size of 0 to 4,294,967,295 could have it’s size returned in a value of type `std::size_t`, so this is fine. However, if the byte-size of an object were larger than 4,294,967,295 bytes, then `sizeof` would not be able to return the size of that object accurately, as the value would be outside the range of a `std::size_t`. Therefore, no object larger than 4,294,967,295 bytes could be created on this system.

As an aside…

The size of `std::size_t` imposes a strict mathematical upper limit on an object’s size. In practice, the largest creatable object may be smaller than this amount (perhaps significantly so).

Some compilers limit the largest creatable object to half the maximum value of `std::size_t` (an explanation for this can be found [here](https://stackoverflow.com/a/42428240)).

Other factors may also play a role, such as how much contiguous memory your computer has available for allocation.

When 8-bit and 16-bit applications were the norm, this limit imposed a significant constraint on the size of objects. In the 32-bit and 64-bit era, this is rarely an issue, and therefore not something you generally need to worry about.

# Floating point numbers

Integers are great for counting whole numbers, but sometimes we need to store _very_ large (positive or negative) numbers, or numbers with a fractional component. A **floating point** type variable is a variable that can hold a number with a fractional component, such as 4320.0, -3.33, or 0.01226. The _floating_ part of the name _floating point_ refers to the fact that the decimal point can “float” -- that is, it can support a variable number of digits before and after the decimal point. Floating point data types are always signed (can hold positive and negative values).



C++ floating point types

C++ has three fundamental floating point data types: a single-precision `float`, a double-precision `double`, and an extended-precision `long double`. As with integers, C++ does not define the actual size of these types.

|Category|C++ Type|Typical Size|
|---|---|---|
|floating point|float|4 bytes|
||double|8 bytes|
||long double|8, 12, or 16 bytes|


On modern architectures, floating-point types are conventionally implemented using one of the floating-point formats defined in the IEEE 754 standard (see [https://en.wikipedia.org/wiki/IEEE_754](https://en.wikipedia.org/wiki/IEEE_754)). As a result, `float` is almost always 4 bytes, and `double` is almost always 8 bytes.


> On the other hand, `long double` is a strange type. On different platforms, its size can vary between 8 and 16 bytes, and it may or may not use an IEEE 754 compliant format. We recommend avoiding `long double`.

Tip

This tutorial series assumes your compiler is using an IEEE 754 compatible format for `float` and `double`.

You can see if your floating point types are IEEE 754 compatible with the following code:

```cpp
#include <iostream>
#include <limits>

int main()
{
    std::cout << std::boolalpha; // print bool as true or false rather than 1 or 0
    std::cout << "float: " << std::numeric_limits<float>::is_iec559 << '\n';
    std::cout << "double: " << std::numeric_limits<double>::is_iec559 << '\n';
    std::cout << "long double: " << std::numeric_limits<long double>::is_iec559 << '\n';
}
```


For advanced readers

`float` is almost always implemented using the 4-byte IEEE 754 single-precision format.  
`double` is almost always implemented using the 8-byte IEEE 754 double-precision format.

However, the format used to implement `long double` varies by platform. Common choices include:

- 8-byte IEEE 754 double-precision format (same as `double`).
- 80-bit (often padded to 12 or 16 bytes) x87 extended-precision format (compatible with IEEE 754).
- 16-byte IEEE 754 quadruple-precision format.
- 16-byte double-double format (not compatible with IEEE 754).



When using floating point literals, always include at least one decimal place (even if the decimal is 0). This helps the compiler understand that the number is a floating point number and not an integer.

```cpp
int a { 5 };      // 5 means integer
double b { 5.0 }; // 5.0 is a floating point literal (no suffix means double type by default)
float c { 5.0f }; // 5.0 is a floating point literal, f suffix means float type

int d { 0 };      // 0 is an integer
double e { 0.0 }; // 0.0 is a double
```

Note that by default, floating point literals default to type double. An `f` suffix is used to denote a literal of type float.

Best practice

Always make sure the type of your literals match the type of the variables they’re being assigned to or used to initialize. Otherwise an unnecessary conversion will result, possibly with a loss of precision.


Floating point range

|Format|Range|Precision|
|---|---|---|
|IEEE 754 single-precision (4 bytes)|±1.18 x 10-38 to ±3.4 x 1038 and 0.0|6-9 significant digits, typically 7|
|IEEE 754 double-precision (8 bytes)|±2.23 x 10-308 to ±1.80 x 10308 and 0.0|15-18 significant digits, typically 16|
|x87 extended-precision (80 bits)|±3.36 x 10-4932 to ±1.18 x 104932 and 0.0|18-21 significant digits|
|IEEE 754 quadruple-precision (16 bytes)|±3.36 x 10-4932 to ±1.18 x 104932 and 0.0|33-36 significant digits|

Floating point precision

Consider the fraction 1/3. The decimal representation of this number is 0.33333333333333… with 3’s going out to infinity. If you were writing this number on a piece of paper, your arm would get tired at some point, and you’d eventually stop writing. And the number you were left with would be close to 0.3333333333…. (with 3’s going out to infinity) but not exactly.

On a computer, an infinite precision number would require infinite memory to store, and we typically only have 4 or 8 bytes per value. This limited memory means floating point numbers can only store a certain number of significant digits -- any additional significant digits are either lost or represented imprecisely. The number that is actually stored may be close to the desired number, but not exact. We’ll show an example of this in the next section.

The **precision** of a floating point type defines how many significant digits it can represent without information loss.

The number of digits of precision a floating point type has depends on both the size (floats have less precision than doubles) and the particular value being stored (some values can be represented more precisely than others).

For example, a float has 6 to 9 digits of precision. This means that a float can exactly represent any number with up to 6 significant digits. A number with 7 to 9 significant digits may or may not be represented exactly depending on the specific value. And a number with more than 9 digits of precision will definitely not be represented exactly.

Double values have between 15 and 18 digits of precision, with most double values having at least 16 significant digits. Long double has a minimum precision of 15, 18, or 33 significant digits depending on how many bytes it occupies.

Key insight

A floating point type can only precisely represent a certain number of significant digits. Using a value with more significant digits than the minimum may result in the value being stored inexactly.


Also note that std::cout will switch to outputting numbers in scientific notation in some cases. Depending on the compiler, the exponent will typically be padded to a minimum number of digits. Fear not, 9.87654e+006 is the same as 9.87654e6, just with some padding 0’s. The minimum number of exponent digits displayed is compiler-specific (Visual Studio uses 3, some others use 2 as per the C99 standard).

We can override the default precision that std::cout shows by using an `output manipulator` function named `std::setprecision()`. **Output manipulators** alter how data is output, and are defined in the _iomanip_ header.

```cpp
#include <iomanip> // for output manipulator std::setprecision()
#include <iostream>

int main()
{
    std::cout << std::setprecision(17); // show 17 digits of precision
    std::cout << 3.33333333333333333333333333333333333333f <<'\n'; // f suffix means float
    std::cout << 3.33333333333333333333333333333333333333 << '\n'; // no suffix means double

    return 0;
}
```

Outputs:

3.3333332538604736
3.3333333333333335

Because we set the precision to 17 digits using `std::setprecision()`, each of the above numbers is printed with 17 digits. But, as you can see, the numbers certainly aren’t precise to 17 digits! And because floats are less precise than doubles, the float has more error.

Tip

Output manipulators (and input manipulators) are sticky -- meaning if you set them, they will remain set.

The one exception is `std::setw`. Some IO operations reset `std::setw`, so `std::setw` should be used every time it is needed.



Precision issues don’t just impact fractional numbers, they impact any number with too many significant digits. Let’s consider a big number:

```cpp
#include <iomanip> // for std::setprecision()
#include <iostream>

int main()
{
    float f { 123456789.0f }; // f has 10 significant digits
    std::cout << std::setprecision(9); // to show 9 digits in f
    std::cout << f << '\n';

    return 0;
}
```

Output:

123456792

123456792 is greater than 123456789. The value 123456789.0 has 10 significant digits, but float values typically have 7 digits of precision (and the result of 123456792 is precise only to 7 significant digits). We lost some precision! When precision is lost because a number can’t be stored precisely, this is called a **rounding error**.

Consequently, one has to be careful when using floating point numbers that require more precision than the variables can hold.


Best practice

Favor double over float unless space is at a premium, as the lack of precision in a float will often lead to inaccuracies.

Rounding errors make floating point comparisons tricky

Floating point numbers are tricky to work with due to non-obvious differences between binary (how data is stored) and decimal (how we think) numbers. Consider the fraction 1/10. In decimal, this is easily represented as 0.1, and we are used to thinking of 0.1 as an easily representable number with 1 significant digit. However, in binary, decimal value 0.1 is represented by the infinite sequence: 0.00011001100110011… Because of this, when we assign 0.1 to a floating point number, we’ll run into precision problems.

You can see the effects of this in the following program:

```cpp
#include <iomanip> // for std::setprecision()
#include <iostream>

int main()
{
    double d{0.1};
    std::cout << d << '\n'; // use default cout precision of 6
    std::cout << std::setprecision(17);
    std::cout << d << '\n';

    return 0;
}
```

This outputs:

0.1
0.10000000000000001

On the top line, `std::cout` prints 0.1, as we expect.

On the bottom line, where we have `std::cout` show us 17 digits of precision, we see that `d` is actually _not quite_ 0.1! This is because the double had to truncate the approximation due to its limited memory. The result is a number that is precise to 16 significant digits (which type double guarantees), but the number is not _exactly_ 0.1. Rounding errors may make a number either slightly smaller or slightly larger, depending on where the truncation happens.


NaN and Inf

IEEE 754 compatible formats additionally support some special values:

- **Inf**, which represents infinity. Inf is signed, and can be positive (+Inf) or negative (-Inf).
- **NaN**, which stands for “Not a Number”. There are several different kinds of NaN (which we won’t discuss here).
- Signed zero, meaning there are separate representations for “positive zero” (+0.0) and “negative zero” (-0.0).

Formats that are not compatible with IEEE 754 may not support some (or any) of these values. In such cases, code that uses or generates these special values will produce implementation-defined behavior.

Here’s a program showing all three:

```cpp
#include <iostream>

int main()
{
    double zero { 0.0 };

    double posinf { 5.0 / zero }; // positive infinity
    std::cout << posinf << '\n';

    double neginf { -5.0 / zero }; // negative infinity
    std::cout << neginf << '\n';

    double z1 { 0.0 / posinf }; // positive zero
    std::cout << z1 << '\n';

    double z2 { -0.0 / posinf }; // negative zero
    std::cout << z2 << '\n';

    double nan { zero / zero }; // not a number (mathematically invalid)
    std::cout << nan << '\n';

    return 0;
}
```

And the results using Clang:

inf
-inf
0
-0
nan

Note that the results of printing `Inf` and `NaN` are platform specific, so your results may vary (e.g. Visual Studio prints the last result as `-nan(ind)`).

Best practice

Avoid division by `0.0`, even if your compiler supports it.


Use `std::boolalpha` to print `true` or `false`

If you want `std::cout` to print `true` or `false` instead of `0` or `1`, you can output `std::boolalpha`. This doesn’t output anything, but manipulates the way `std::cout` outputs `bool` values.

Here’s an example:

```cpp
#include <iostream>

int main()
{
    std::cout << true << '\n';
    std::cout << false << '\n';

    std::cout << std::boolalpha; // print bools as true or false

    std::cout << true << '\n';
    std::cout << false << '\n';
    return 0;
}
```

You can use `std::noboolalpha` to turn it back off.


Explain this:
Integer to Boolean conversion

When using uniform initialization, you can initialize a variable using integer literals 0 (for false) and 1 (for true) (but you really should be using false and true instead). Other integer literals cause compilation errors:
```cpp

#include <iostream>

int main()
{
	bool bFalse { 0 }; // okay: initialized to false
	bool bTrue  { 1 }; // okay: initialized to true
	bool bNo    { 2 }; // error: narrowing conversions disallowed

	std::cout << bFalse << bTrue << bNo << '\n';

	return 0;
}
```

```cpp
However, in any context where an integer can be converted to a Boolean, the integer 0 is converted to false, and any other integer is converted to true.

#include <iostream>

int main()
{
	std::cout << std::boolalpha; // print bools as true or false

	bool b1 = 4 ; // copy initialization allows implicit conversion from int to bool
	std::cout << b1 << '\n';

	bool b2 = 0 ; // copy initialization allows implicit conversion from int to bool
	std::cout << b2 << '\n';

	return 0;
}
```

This prints:

true
false

Note: bool b1 = 4; may generate a warning. If so you’ll have to disable treating warnings as errors to compile the example.

Inputting Boolean values

Inputting Boolean values using std::cin sometimes trips new programmers up.

Consider the following program:
```cpp

#include <iostream>

int main()
{
	bool b{}; // default initialize to false
	std::cout << "Enter a boolean value: ";
	std::cin >> b;
	std::cout << "You entered: " << b << '\n';

	return 0;
}
```

Enter a Boolean value: true
You entered: 0

Wait, what?

By default, std::cin only accepts numeric input for Boolean variables: 0 is false, and 1 is true. Any other numeric value will be interpreted as true, and will cause std::cin to enter failure mode. Any non-numeric value will be interpreted as false and will cause std::cin to enter failure mode.

Related content

We discuss failure mode (and how to get out of it) in lesson 9.5 -- std::cin and handling invalid input.


Short version:

**Initialization**

* `{}` (uniform init): only `0` → `false`, `1` → `true`. Anything else = compile error (prevents bugs).
* `=` (copy init): `0` → `false`, **any non-zero** → `true`.

**Why**

* Uniform init forbids narrowing.
* Copy init allows implicit int→bool conversion.

**Conversion rule**

* `0` → `false`
* non-zero → `true`

**Input with `std::cin`**

* `0` → `false`
* `1` → `true`
* other numbers → `true` **but** stream fails
* non-numbers (`true`, `false`) → `false` **and** stream fails

**Key takeaway**

* Use `true` / `false` in code.
* Don’t expect `cin` to accept `"true"` unless you enable `std::boolalpha`.


It means **`std::cin` goes into an error state**.

When the stream fails:

* The input operation stops.
* The variable may be unchanged or set to `false`.
* All future `cin >> ...` do nothing until fixed.

Specifically:

* `cin.fail()` becomes `true`.
* The bad input stays in the buffer.

Example:

```cpp
bool b;
std::cin >> b;   // enter: true
```

Result:

* `b == false`
* `cin.fail() == true`

To recover:

```cpp
std::cin.clear();            // reset error flags
std::cin.ignore(1000, '\n'); // discard bad input
```

In short: **stream fails = input is broken until you clear it.**


To allow `std::cin` to accept the words `false` and `true` as inputs, you must first input to `std::boolalpha`:


However, when `std::boolalpha` is enabled for input, numeric values will no longer be accepted (they evaluate to `false` and cause std::cin to enter failure mode).


Warning

Enabling `std::boolalpha` for input will only allow lower-cased `false` or `true` to be accepted. Variations with capital letters will not be accepted. `0` and `1` will also no longer be accepted.


Note that we use `std::cin >> std::boolalpha;` to input bool values as `true` or `false`, and `std::cout << std::boolalpha;` to output bool values as `true` or `false`. These are independent controls that can be turned on (using `std::boolalpha`) or off (using `std::noboolalpha`) separately.


## Chars

The char data type is an integral type, meaning the underlying value is stored as an integer. Similar to how a Boolean value `0` is interpreted as `false` and non-zero is interpreted as `true`, the integer stored by a `char` variable are intepreted as an `ASCII character`.

**ASCII** stands for American Standard Code for Information Interchange, and it defines a particular way to represent English characters (plus a few other symbols) as numbers between 0 and 127 (called an **ASCII code** or **code point**). For example, ASCII code 97 is interpreted as the character ‘a’.


Codes 0-31 and 127 are called the unprintable chars. These codes were designed to control peripheral devices such as printers (e.g. by instructing the printer how to move the print head). Most of these are obsolete now. If you try to print these chars, the results are dependent upon your OS (you may get some emoji-like characters).

Codes 32-126 are called the printable characters, and they represent the letters, number characters, and punctuation that most computers use to display basic English text.

If you try to print a character whose value is outside the range of ASCII, the results are also dependent upon your OS.


Initializing chars

You can initialize char variables using character literals:

```cpp
char ch2{ 'a' }; // initialize with code point for 'a' (stored as integer 97) (preferred)
```

You can initialize chars with integers as well, but this should be avoided if possible

```cpp
char ch1{ 97 }; // initialize with integer 97 ('a') (not preferred)
```

Warning

Be careful not to mix up character numbers with integer numbers. The following two initializations are not the same:

```cpp
char ch{5}; // initialize with integer 5 (stored as integer 5)
char ch{'5'}; // initialize with code point for '5' (stored as integer 53)
```

Character numbers are intended to be used when we want to represent numbers as text, rather than as numbers to apply mathematical operations to.


Printing chars

When using std::cout to print a char, std::cout outputs the char variable as an ASCII character:

```cpp
#include <iostream>

int main()
{
    char ch1{ 'a' }; // (preferred)
    std::cout << ch1; // cout prints character 'a'

    char ch2{ 98 }; // code point for 'b' (not preferred)
    std::cout << ch2; // cout prints a character ('b')


    return 0;
}
```


Note that std::cin will let you enter multiple characters. However, variable _ch_ can only hold 1 character. Consequently, only the first input character is extracted into variable _ch_. The rest of the user input is left in the input buffer that std::cin uses, and can be extracted with subsequent calls to std::cin.

If you want to read in more than one char at a time (e.g. to read in a name, word, or sentence), you’ll want to use a string instead of a char. A string is a collection of sequential characters (and thus, a string can hold multiple symbols). We discuss this in upcoming lesson ([5.7 -- Introduction to std::string](https://www.learncpp.com/cpp-tutorial/introduction-to-stdstring/)).

Extracting whitespace characters

Because extracting input ignores leading whitespace, this can lead to unexpected results when trying to extract whitespace characters to a char variable:

```cpp
#include <iostream>

int main()
{
    std::cout << "Input a keyboard character: "; // assume the user enters "a b" (without quotes)

    char ch{};
    std::cin >> ch; // extracts a, leaves " b\n" in stream
    std::cout << "You entered: " << ch << '\n';

    std::cin >> ch; // skips leading whitespace (the space), extracts b, leaves "\n" in stream
    std::cout << "You entered: " << ch << '\n';

    return 0;
}
```

Input a keyboard character: a b
You entered: a
You entered: b

In the above example, we may have expected to extract the space, but because leading whitespace is skipped, we extracted the `b` character instead.

One simple way to address this is to use the `std::cin.get()` function to perform the extraction instead, as this function does not ignore leading whitespace:

```cpp
#include <iostream>

int main()
{
    std::cout << "Input a keyboard character: "; // assume the user enters "a b" (without quotes)

    char ch{};
    std::cin.get(ch); // extracts a, leaves " b\n" in stream
    std::cout << "You entered: " << ch << '\n';

    std::cin.get(ch); // extracts space, leaves "b\n" in stream
    std::cout << "You entered: " << ch << '\n';

    return 0;
}
```

Input a keyboard character: a b
You entered: a
You entered:


Char size, range, and default sign

Char is defined by C++ to always be 1 byte in size. By default, a char may be signed or unsigned (though it’s usually signed). If you’re using chars to hold ASCII characters, you don’t need to specify a sign (since both signed and unsigned chars can hold values between 0 and 127).

If you’re using a char to hold small integers (something you should not do unless you’re explicitly optimizing for space), you should always specify whether it is signed or unsigned. A signed char can hold a number between -128 and 127. An unsigned char can hold a number between 0 and 255.


Escape sequences

There are some sequences of characters in C++ that have special meaning. These characters are called **escape sequences**. An escape sequence starts with a ‘\’ (backslash) character, and then a following letter or number.

You’ve already seen the most common escape sequence: `'\n'`, which can be used to print a newline:


Another commonly used escape sequence is `'\t'`, which embeds a horizontal tab:


Three other notable escape sequences are:  
\’ prints a single quote  
\” prints a double quote  
\\ prints a backslash

Here’s a table of all of the escape sequences:

|Name|Symbol|Meaning|
|---|---|---|
|Alert|\a|Makes an alert, such as a beep|
|Backspace|\b|Moves the cursor back one space|
|Formfeed|\f|Moves the cursor to next logical page|
|Newline|\n|Moves cursor to next line|
|Carriage return|\r|Moves cursor to beginning of line|
|Horizontal tab|\t|Prints a horizontal tab|
|Vertical tab|\v|Prints a vertical tab|
|Single quote|\’|Prints a single quote|
|Double quote|\”|Prints a double quote|
|Backslash|\\|Prints a backslash.|
|Question mark|\?|Prints a question mark.  <br>No longer relevant. You can use question marks unescaped.|
|Octal number|\(number)|Translates into char represented by octal|
|Hex number|\x(number)|Translates into char represented by hex number|

What’s the difference between putting symbols in single and double quotes?

Text between single quotes is treated as a `char` literal, which represents a single character. For example, `'a'` represents the character `a`, `'+'` represents the character for the plus symbol, `'5'` represents the character `5` (not the number 5), and `'\n'` represents the newline character.

Text between double quotes (e.g. “Hello, world!”) is treated as a C-style string literal, which can contain multiple characters. We discuss strings in lesson [5.2 -- Literals](https://www.learncpp.com/cpp-tutorial/literals/).

Best practice

Single characters should usually be single-quoted, not double-quoted (e.g. `'t'` or `'\n'`, not `"t"` or `"\n"`). One possible exception occurs when doing output, where it can be preferential to double quote everything for consistency (see lesson [1.5 -- Introduction to iostream: cout, cin, and endl](https://www.learncpp.com/cpp-tutorial/introduction-to-iostream-cout-cin-and-endl/)).


Avoid multicharacter literals

For backwards compatibility reasons, many C++ compilers support **multicharacter literals**, which are char literals that contain multiple characters (e.g. `'56'`). If supported, these have an implementation-defined value (meaning it varies depending on the compiler). Because they are not part of the C++ standard, and their value is not strictly defined, multicharacter literals should be avoided.

Best practice

Avoid multicharacter literals (e.g. `'56'`).


What about the other char types, `wchar_t`, `char8_t`, `char16_t`, and `char32_t`?

Much like ASCII maps the integers 0-127 to American English characters, other character encoding standards exist to map integers (of varying sizes) to characters in other languages. The most well-known mapping outside of ASCII is the Unicode standard, which maps over 144,000 integers to characters in many different languages. Because Unicode contains so many code points, a single Unicode code point needs 32-bits to represent a character (called UTF-32). However, Unicode characters can also be encoded using multiple 16-bit or 8-bit characters (called UTF-16 and UTF-8 respectively).

`char16_t` and `char32_t` were added to C++11 to provide explicit support for 16-bit and 32-bit Unicode characters. These char types have the same size as `std::uint_least16_t` and `std::uint_least32_t` respectively (but are distinct types). `char8_t` was added in C++20 to provide support for 8-bit Unicode (UTF-8). It is a distinct type that uses the same representation as `unsigned char`.

You won’t need to use `char8_t`, `char16_t`, or `char32_t` unless you’re planning on making your program Unicode compatible. `wchar_t` should be avoided in almost all cases (except when interfacing with the Windows API), as its size is implementation-defined.

Unicode and localization are generally outside the scope of these tutorials, so we won’t cover it further. In the meantime, you should only use ASCII characters when working with characters (and strings). Using characters from other character sets may cause your characters to display incorrectly.


## Introduction to type conversion and static_cast


When the compiler does type conversion on our behalf without us explicitly asking, we call this **implicit type conversion**. The above example illustrates this -- nowhere do we explicitly tell the compiler to convert integer value `5` to double value `5.0`. Rather, the function is expecting a double value, and we pass in an integer argument. The compiler will notice the mismatch and implicitly convert the integer to a double.

Type conversion of a value produces a new value

The type conversion process does not modify the value (or object) supplying the data to be converted. Instead, the conversion process uses that data as input, and produces the converted result.

Key insight

The type conversion of a value to another type of value behaves much like a call to a function whose return type matches the target type of the conversion. The data to be converted is passed in as an argument, and the converted result is returned (in a temporary object) to be used by the caller.

In the above example, the conversion does not change variable `y` from type `int` to `double` or the value of `y` from `5` to `5.0`. Instead, the conversion uses the value of `y` (`5`) as input, and returns a temporary object of type `double` with value `5.0`. This temporary object is then passed to function `print`.

For advanced readers

Some advanced type conversions (e.g. those involving `const_cast` or `reinterpret_cast`) do not return temporary objects, but instead reinterpret the type of an existing value or object.


In this program, we’ve changed `print()` to take an `int` parameter, and the function call to `print()` is now passing in `double` value `5.5`. Similar to the above, the compiler will use implicit type conversion in order to convert double value `5.5` into a value of type `int`, so that it can be passed to function `print()`.

Unlike the initial example, when this program is compiled, your compiler will generate some kind of a warning about a possible loss of data. And because you have “treat warnings as errors” turned on (you do, right?), your compiler will abort the compilation process.


Because converting a floating point value to an integral value results in any fractional component being dropped, the compiler will warn us when it does an implicit type conversion from a floating point to an integral value. This happens even if we were to pass in a floating point value with no fractional component, like `5.0` -- no actual loss of value occurs during the conversion to integral value `5` in this specific case, but the compiler may still warn us that the conversion is unsafe.


C++ supports a second method of type conversion, called explicit type conversion. **Explicit type conversion** allow us (the programmer) to explicitly tell the compiler to convert a value from one type to another type, and that we take full responsibility for the result of that conversion. If such a conversion results in the loss of value, the compiler will not warn us.

To perform an explicit type conversion, in most cases we’ll use the `static_cast` operator. The syntax for the `static cast` looks a little funny:

static_cast<new_type>(expression)

static_cast takes the value from an expression as input, and returns that value converted into the type specified by _new_type_ (e.g. int, bool, char, double).

Key insight

Whenever you see C++ syntax (excluding the preprocessor) that makes use of angled brackets (\<>), the thing between the angled brackets will most likely be a type. This is typically how C++ deals with code that need a parameterized type.



```cpp
int main()
{
	print( static_cast<int>(5.5) ); // explicitly convert double value 5.5 to an int

	return 0;
}
```


It’s worth noting that the argument to _static_cast_ evaluates as an expression. When we pass in a variable, that variable is evaluated to produce its value, and that value is then converted to the new type. The variable itself is _not_ affected by casting its value to a new type. In the above case, variable `ch` is still a char, and still holds the same value even after we’ve cast its value to an `int`.


Sign conversions using static_cast


If the value being converted cannot be represented in the destination type:

- If the destination type is unsigned, the value will be modulo wrapped. We cover modulo wrapping in lesson [4.5 -- Unsigned integers, and why to avoid them](https://www.learncpp.com/cpp-tutorial/unsigned-integers-and-why-to-avoid-them/).
- If the destination type is signed, the value is implementation-defined prior to C++20, and will be modulo wrapped as of C++20.

Here’s an example of converting two values that are not representable in the destination type (assuming 32-bit integers):

```cpp
#include <iostream>

int main()
{
    int s { -1 };
    std::cout << static_cast<unsigned int>(s) << '\n'; // prints 4294967295

    unsigned int u { 4294967295 }; // largest 32-bit unsigned int
    std::cout << static_cast<int>(u) << '\n'; // implementation-defined prior to C++20, -1 as of C++20

    return 0;
}
```

As of C++20, this produces the result:

4294967295
-1

Signed int value `-1` cannot be represented as an unsigned int. The result modulo wraps to unsigned int value `4294967295`.

Unsigned int value `4294967295` cannot be represented as a signed int. Prior to C++20, the result is implementation defined (but will probably be `-1`). As of C++20, the result will modulo wrap to `-1`.

Warning

Converting an unsigned integral value to a signed integral value will result in implementation-defined behavior prior to C++20 if the value being converted can not be represented in the signed type.


