## Primitives and references
Primitives can be copied directly whereas references cannot
```ts
var a = 12;
var b = a;
b = 2;
console.log(b);
// 2

var a = [12];
var b = a;
b[0] = 2;
console.log(a[0]);
// 2
```
### Primitives
Numbers, strings and booleans

### References
Arrays, tuples, enums

## Arrays
```ts
let arr = [1,2,3, 'harsh']; // automatically sets array type to string and number

let arr = [1,2,3, {name: 'harsh'}, {name: 123}]; 
// type is now number and object with name as string and name as number

let arr: number[] = [1,2,3, {name: 'harsh'}, {name: 123}];
// it will give error now, it will accept only numbers

let arr: number[] = [1,2,3];
// alright now
```

## Tuples
```ts
let arr: [string, number] = ["abc", 12];
// array of fixed size and fixed type
```

## Enums
```ts
enum userRoles {
    admin = 'admin1',
    user = 'user1',
    super_admin = 'super1'
}

console.log(userRoles.admin)
```

## Any
```ts
let a:number;
a=12;

let b:any; // or let b; it automatically sets type to any
b='bhavneek';
b=12;
```

## Unknown
```ts
let c:unknown;
c=12;
c="abc";

if(typeof c === "string"){
    c.toUpperCase();
}
```
Unknown does not let use type specific functions to the variable like c.toUpperCase() unless there is a check used like ==typeof== 

## Void
```ts
function hey(): void{
    console.log("hey")
}
```

## Null
```ts
let d : string | null;
```
in case return from a DB comes out as null

## Never
A function returning type ==never== will never let code run forward

# Inference and annotations
Using : to define type is annotation and typescript automatically inferencing type from init value is inference

