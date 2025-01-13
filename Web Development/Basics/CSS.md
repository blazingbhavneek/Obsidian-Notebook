Cheat sheet: https://adam-marsden.co.uk/css-cheat-sheet
# Including CSS
Ways to include css
1. External file using ".css" extension, called using link tag
2. Internal CSS (using <style></style> tag)
3. Inline css (using prop style = "...")
Order of precedence is external/internal (depending on which is called first) < inline css

# Selectors
```css
tag_type {}
.class_name {}
#element_id {}
```

> concatenating tag-types (`tag1 tag2 {}`) means all tag2's in tag1, while using comma means these two tags have same style
> Precedance of selector: tags < class < id
> To overcome precedance write ==!important== after property

## Inheritance
A property defined in a parent class will only go to its children if it CAN inherit the property, for example, defining font size in body tag will be inherited by text in other tags, but border will be not
> unless we select all the tags using \*{border: 2px red}

# Text Properties
```css
color
font-style
font-size
text-transform
background-color
line-height

```

# Colors
CSS recognizes 140 color names
or we can use custom color using 
- `color: rgba(r, g, b, t)` (t is transparency)
- hexadecimal coding: `color: #FFFFFF`
Use color picker extension/websites to get good color palletes
https://coolors.co/ (Also for color contrast checker for accessability)

# Units and Sizes
- ==Avoid using absolute sizes for font (pixels, centimeters)== as it can override user preference setting for small or large text
- Using percentage for width/ other dimensions takes the size of parent for reference
- For text, use `__rem` for font-size, where __ is a number. It sets font size relative to root element font size which is set by browser and client prefernce. 
- `em` takes value of `font-size` of the same element, and if it doesn't have `font-size` then it will take font-size of nearest parent for reference); is mostly used for properties like width, etc
- `ch` suffix for size stands for characters, width of the text box sets according to number of characters 
- `vw / vh` viewport width/height % of width of screen 
	- ==take care using `vw` because it may create a scrollbar if contents exceed a certain limit and that will induce a horizontal scrollbar==

# Box model
An element has 3 parts
- content
- padding
- border
- outline (optional)
- margin


When we set width of a box, we can change what width actually means using below
```css
* {
 box-sizing: border-box | content-box;
}
```
Width with Border box = Content + padding + border
Width with Content box = Only content

# Display
1. display: block
	1. Takes up the full width available.
	2. Starts on a new line.
	3. Examples: `<div>, <p>, <h1>`.
2. display: inline
	1. Takes up only as much width as its content.
	2. Does not start on a new line.
	3. Examples: `<span>, <a>, <strong>`.
3. display: inline-block
	1. Behaves like inline but allows setting width/height.
	2. Does not start on a new line.

# CSS float Property:
1. Purpose: Used to place elements on the left or right, allowing text or inline elements to wrap around it.
2. Values:
	1. left: Floats the element to the left.
	2. right: Floats the element to the right.
	3. none: Default; no float.
	4. inherit: Inherits from the parent.
3. Common Use Cases:
	1. Creating layouts (before flexbox and grid).
	2. Wrapping text around images.

## Common Pitfalls & Problems:
- Collapsing Parent Height
	- Floated elements don’t contribute to the height of their container.
		- Solution: Use clearfix or overflow: hidden on the parent. (modern:`display: flow-root` on parent container)
- Content Overlap:
	- Non-floated content can overlap floated elements.
	- Solution: Use clear: both to prevent overlap.
- Float Drop:
	- When there isn’t enough horizontal space, floats drop down unexpectedly.
	- Solution: Adjust the width or use media queries.
- Over-reliance on Floats:
	- Overusing floats for layout can lead to brittle designs.
	- Solution: Prefer modern layout techniques (flexbox, grid).
- Clearfix Hack:
	- Necessary to fix parent height issues when using floats.

```css
.clearfix::after {
    content: "";
    display: block;
    clear: both;
}
```

# Positions
1. Absolute
	1. An element with absolute position needs a parent with position property relative, or else it will take the "main" or the whole page as a parent
2. Relative
	1. Is always going to be relative to its parent container regardless of whether is parent is relative or not
3. Fixed
	1. Will stay fixed relative to browser window, will move along with the scroll
4. Sticky
	1. Tries to stay on screen as much as possible when scrolled, boundary is defined by its parent container. When going out of screen because of scroll, it starts to move

# Flexbox
https://flexbox.malven.co/
Visual cheatsheet

# Grids
https://grid.layoutit.com/
Interactive cheat sheet

## Properties
```css
.container{
display: grid;
grid-auto-flow: row | column;
grid-template-columns: 100px 1fr 2fr 1fr 2fr; /* also check out other available units in cheatsheet */
grid-template-rows: 100px repeat(2, 1fr 1fr); /* both rows and columns are same above */
row-gap: 1em;
column-gap: 1rem;

}
```
# Convert CSS to Tailwind
Cheatsheet
https://nerdcave.com/tailwind-cheat-sheet

Converter
https://tailwind-converter.netlify.app/