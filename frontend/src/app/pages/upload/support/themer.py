import sys
import re

def remove_comments(css: str) -> str:
    # Remove /* ... */ comments, including newlines
    return re.sub(r'/\*.*?\*/', '', css, flags=re.DOTALL)

def find_matching_brace(text: str, start_index: int) -> int:
    """Given text and the index of a '{', find the matching '}' index. Returns -1 if not found."""
    level = 0
    i = start_index
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '{':
            level += 1
        elif ch == '}':
            level -= 1
            if level == 0:
                return i
        i += 1
    return -1

def split_selectors(selector_text: str):
    """
    Split a selector list on commas, while not breaking on commas inside brackets/parens.
    This is a simplified splitter that handles most common cases.
    """
    parts = []
    buf = []
    level_round = 0
    level_square = 0
    level_curly = 0  # shouldn't appear in selector, but be safe
    for ch in selector_text:
        if ch == '(':
            level_round += 1
        elif ch == ')':
            level_round = max(0, level_round - 1)
        elif ch == '[':
            level_square += 1
        elif ch == ']':
            level_square = max(0, level_square - 1)
        elif ch == '{':
            level_curly += 1
        elif ch == '}':
            level_curly = max(0, level_curly - 1)
        elif ch == ',' and level_round == 0 and level_square == 0 and level_curly == 0:
            parts.append(''.join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    # Drop empty selectors defensively
    return [p for p in parts if p]

def is_at_rule(header: str) -> bool:
    return header.lstrip().startswith('@')

def is_keyframes_rule(header: str) -> bool:
    return header.lstrip().lower().startswith('@keyframes')

def prefix_selectors(header: str, theme_class: str) -> str:
    """
    Prefix each selector with :host(.theme_class).
    Example:
      header: ".a, .b:hover"
      => ":host(.theme) .a, :host(.theme) .b:hover"
    """
    selectors = split_selectors(header)
    prefixed = []
    for s in selectors:
        if not s:
            continue
        prefixed.append(f":host(.{theme_class}) {s}")
    return ", ".join(prefixed)

def process_block(css: str, start: int, theme_class: str) -> (str, int):
    """
    Processes a sequence of rules starting at index `start` until EOF or an unmatched '}'.
    Returns (processed_text, new_index).
    """
    i = start
    n = len(css)
    out = []

    while i < n:
        # Skip whitespace
        while i < n and css[i].isspace():
            out.append(css[i])
            i += 1
        if i >= n:
            break

        # If we hit a closing brace, return to caller (end of this block)
        if css[i] == '}':
            return ''.join(out), i

        # Read header up to next '{' or ';'
        header_start = i
        brace_pos = css.find('{', i)
        semi_pos = css.find(';', i)

        if brace_pos == -1 and semi_pos == -1:
            # No more rules; append remainder and break
            out.append(css[i:])
            i = n
            break

        # If there's a ; before a {, it's an at-rule without a block (e.g., @import)
        if semi_pos != -1 and (brace_pos == -1 or semi_pos < brace_pos):
            header = css[header_start:semi_pos].rstrip()
            # Emit at-rule without any change (like @import url(...);)
            out.append(header)
            out.append(';')
            i = semi_pos + 1
            continue

        # Otherwise we have a block: header { ... }
        header = css[header_start:brace_pos].strip()
        # Append any whitespace between header and '{' exactly as-is
        between = css[brace_pos:brace_pos+1]  # this is '{'
        # Find matching closing brace for this block
        block_end = find_matching_brace(css, brace_pos)
        if block_end == -1:
            # Malformed CSS; emit the rest and bail
            out.append(css[i:])
            i = n
            break

        inner_start = brace_pos + 1
        inner_end = block_end  # exclusive
        inner_content = css[inner_start:inner_end]

        if is_keyframes_rule(header):
            # Keep @keyframes completely intact
            out.append(header)
            out.append('{')
            out.append(inner_content)  # untouched
            out.append('}')
        elif is_at_rule(header):
            # Generic at-rule with a block, like @media, @supports, @layer, @container, @font-face
            # For @font-face, we should not prefix properties (no selectors); but it’s okay to process recursively:
            # the recursive call will see declarations (no nested '{') and just reproduce them.
            out.append(header)
            out.append('{')
            processed_inner, _ = process_block(inner_content, 0, theme_class)
            out.append(processed_inner)
            out.append('}')
        else:
            # Normal selector block — prefix selectors
            prefixed_header = prefix_selectors(header, theme_class)
            out.append(prefixed_header)
            out.append('{')
            # Process nested blocks if any (e.g., future nested syntax). Typically declarations only.
            processed_inner, _ = process_block(inner_content, 0, theme_class)
            out.append(processed_inner)
            out.append('}')

        i = block_end + 1

    return ''.join(out), i

def process_css_text(css: str, theme_class: str) -> str:
    # 1) Remove comments first
    css_no_comments = remove_comments(css)
    # 2) Process from the top-level
    processed, _ = process_block(css_no_comments, 0, theme_class)
    return processed

def process_css_file(input_file, output_file, theme_class):
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    processed_content = process_css_text(content, theme_class)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(f"/* --- STYLES FOR THEME: {theme_class} (processed) --- */\n\n")
        f_out.write(processed_content)

    print(f"Successfully processed '{input_file}' -> '{output_file}' for theme '{theme_class}'.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_css.py <input_css_file> <output_css_file> <theme_name>")
        print("Example: python process_css.py styles.css prefixed_styles.css theme-new")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    theme_name = sys.argv[3]

    process_css_file(input_path, output_path, theme_name)
