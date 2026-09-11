// SEE: https://github.com/mathjax/MathJax/issues/2708#issuecomment-861779454
// SEE: https://github.com/orgs/sphinx-doc/discussions/13147#discussioncomment-11837201

/** @type {Map<string, string>} */
const SUPERSCRIPT_MAP = new Map([
    // SEE: https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts
    // --- Digits (note ¹²³ live in Latin-1 Supplement) ---
    ["⁰", "0"], ["¹", "1"], ["²", "2"], ["³", "3"], ["⁴", "4"],
    ["⁵", "5"], ["⁶", "6"], ["⁷", "7"], ["⁸", "8"], ["⁹", "9"],

    // --- Basic operators / punctuation in the Superscripts and Subscripts block ---
    ["⁺", "+"], ["⁻", "-"], ["⁼", "="], ["⁽", "("], ["⁾", ")"],

    // --- Common Latin superscript (modifier) letters (lowercase/minuscule) ---
    // NOTE: As of 2026, all 26 letters are available.
    ["ᵃ", "a"], ["ᵇ", "b"], ["ᶜ", "c"], ["ᵈ", "d"], ["ᵉ", "e"], ["ᶠ", "f"], ["ᵍ", "g"],
    ["ʰ", "h"], ["ⁱ", "i"], ["ʲ", "j"], ["ᵏ", "k"], ["ˡ", "l"], ["ᵐ", "m"], ["ⁿ", "n"],
    ["ᵒ", "o"], ["ᵖ", "p"], ["𐞥", "q"], ["ʳ", "r"], ["ˢ", "s"], ["ᵗ", "t"], ["ᵘ", "u"],
    ["ᵛ", "v"], ["ʷ", "w"], ["ˣ", "x"], ["ʸ", "y"], ["ᶻ", "z"],

    // --- Latin superscript capitals (modifier letters) ---
    // NOTE: As of 2026, only {X, Y, Z} are missing.
    ["ᴬ", "A"], ["ᴮ", "B"], ["ꟲ", "C"], ["ᴰ", "D"], ["ᴱ", "E"], ["ꟳ", "F"], ["ᴳ", "G"],
    ["ᴴ", "H"], ["ᴵ", "I"], ["ᴶ", "J"], ["ᴷ", "K"], ["ᴸ", "L"], ["ᴹ", "M"], ["ᴺ", "N"],
    ["ᴼ", "O"], ["ᴾ", "P"], ["ꟴ", "Q"], ["ᴿ", "R"], ["꟱", "S"], ["ᵀ", "T"], ["ᵁ", "U"],
    ["ⱽ", "V"], ["ᵂ", "W"], /*	    */  /*	    */  /*	    */

    // --- Optional: Greek modifier superscripts (mapped to TeX macros) ---
    ["ᵅ", "\\alpha"], ["ᵝ", "\\beta"], ["ᵞ", "\\gamma"], ["ᵟ", "\\delta"], ["ᵋ", "\\epsilon"],
    ["ᶿ", "\\theta"], ["ᶥ", "\\iota"], ["ᶹ", "\\upsilon"], ["ᵠ", "\\phi"], ["ᵡ", "\\chi"],
]);


/**
 * Convert supported Unicode superscripts in a string into TeX subscripts.
 *
 * @param {string} s - Input string (e.g. MathJax MathItem.math)
 * @returns {string} - Output string with subscripts rewritten as ^{...}
 */
const unicodeSuperscriptsToTeX = (s) => {
    let parts = [];
    let buffer = ""; // string buffer for collected mapped superscripts
    let start = 0; // UTF-16 index of the current slice start
    let i = 0;	 // current UTF-16 index while iterating

    for (const ch of s) {
        const mapped = SUPERSCRIPT_MAP.get(ch);

        if (mapped !== undefined) {  // starting or continuing superscript
            if (!buffer) {
                parts.push(s.slice(start, i));
            }
            buffer += mapped;
        } else if (buffer) {  // ending superscript, flush buffer
            parts.push(`^{${buffer}}`);
            buffer = "";
            start = i;
        }
        i += ch.length;
    }

    parts.push((buffer) ? `^{${buffer}}` : s.slice(start));  // final part
    return parts.join("");
}


/* @type {Map<string, string>} */
const SUBSCRIPT_MAP = new Map([
    // SEE: https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts
    // --- Digits ---
    ["₀", "0"], ["₁", "1"], ["₂", "2"], ["₃", "3"], ["₄", "4"],
    ["₅", "5"], ["₆", "6"], ["₇", "7"], ["₈", "8"], ["₉", "9"],

    // --- Arithmetic / parentheses ---
    ["₊", "+"], ["₋", "-"], ["₌", "="], ["₍", "("], ["₎", ")"],

    // --- Latin lowercase subscripts Unicode actually provides ---
    // NOTE: as of 2026 the following are missing: {b, c, d, f, g, q, w, y, z}
    ["ₐ", "a"], /*	    */  /*	    */  /*	    */  ["ₑ", "e"], /*	    */  /*	    */
    ["ₕ", "h"], ["ᵢ", "i"], ["ⱼ", "j"], ["ₖ", "k"], ["ₗ", "l"], ["ₘ", "m"], ["ₙ", "n"],
    ["ₒ", "o"], ["ₚ", "p"], /*	    */  ["ᵣ", "r"], ["ₛ", "s"], ["ₜ", "t"], ["ᵤ", "u"],
    ["ᵥ", "v"], /*	    */  ["ₓ", "x"], /*	    */  /*	    */

    // --- Greek subscripts Unicode provides (mapped to TeX macros) ---
    ["ᵦ", "\\beta"], ["ᵧ", "\\gamma"], ["ᵨ", "\\rho"], ["ᵩ", "\\phi"], ["ᵪ", "\\chi"],
]);


/**
 * Convert runs of supported Unicode subscripts in a string into TeX subscripts.
 *
 * @param {string} s - Input string (e.g. MathJax MathItem.math)
 * @returns {string} - Output string with subscripts rewritten as _{...}
 */
const unicodeSubscriptsToTeX = (s) => {
    let parts = [];
    let buffer = ""; // string buffer for collected mapped superscripts
    let start = 0; // index of the current slice start
    let i = 0;	 // current index while iterating

    for (const ch of s) {
        const mapped = SUBSCRIPT_MAP.get(ch);

        if (mapped !== undefined) {  // starting or continuing superscript
            if (!buffer) {
                parts.push(s.slice(start, i));
            }
            buffer += mapped;
        } else if (buffer) {  // ending superscript, flush buffer
            parts.push(`_{${buffer}}`);
            buffer = "";
            start = i;
        }
        i += ch.length;
    }

    parts.push((buffer) ? `_{${buffer}}` : s.slice(start));  // final part
    return parts.join("");
};


/** @type {Map<string, string>} */
const LETTER_MODIFIER_MAP = new Map([
    // Unicode combining marks mapped to MathJax accent commands.
    ["\u20D7", "vec"], // U+20D7 COMBINING RIGHT ARROW ABOVE
    ["\u0300", "grave"], // U+0300 COMBINING GRAVE ACCENT
    ["\u0301", "acute"], // U+0301 COMBINING ACUTE ACCENT
    ["\u0302", "hat"], // U+0302 COMBINING CIRCUMFLEX ACCENT
    ["\u0303", "tilde"], // U+0303 COMBINING TILDE
    ["\u0306", "breve"], // U+0306 COMBINING BREVE
    ["\u0307", "dot"], // U+0307 COMBINING DOT ABOVE
    ["\u0308", "ddot"], // U+0308 COMBINING DIAERESIS
    ["\u20DB", "dddot"], // U+20DB COMBINING THREE DOTS ABOVE
    ["\u0304", "bar"], // U+0304 COMBINING MACRON
    ["\u030A", "mathring"], // U+030A COMBINING RING ABOVE
    ["\u030C", "check"], // U+030C COMBINING CARON
]);

// Capture the letter and modifier separately for the replacement callback.
const LETTER_MODIFIER_PATTERN = (
    /([\p{Script=Latin}\p{Script=Greek}])([\u20D7\u0300\u0301\u0302\u0303\u0306\u0307\u0308\u20DB\u0304\u030A\u030C])/gu
);


/**
 * Convert supported Unicode combining letter modifiers into TeX accents.
 *
 * Latin and Greek letters are supported, so, for example, ``β̂`` becomes
 * ``\\hat{β}``. The callback leaves TeX control sequences such as ``\\alphâ``
 * untouched: its combining mark belongs to the control-sequence name rather
 * than to a literal Unicode letter.
 *
 * @param {string} s - Input string (e.g. MathJax MathItem.math)
 * @returns {string} - Output string with modifiers rewritten as TeX accents.
 */
const unicodeLetterModifiersToTeX = (s) => s.replace(
    LETTER_MODIFIER_PATTERN,
    (match, letter, modifier, offset, input) => {
        // Do not treat the final character in a TeX control sequence as a
        // literal letter (e.g. the ``a`` in ``\\alphâ``).
        if (/\\[A-Za-z]*$/.test(input.slice(0, offset))) {
            return match;
        }

        const command = LETTER_MODIFIER_MAP.get(modifier);
        return (command === undefined) ? match : `\\${command}{${letter}}`;
    },
);


// SEE: https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax
// SEE: https://docs.mathjax.org/en/v4.0/advanced/synchronize/filters.html#converting-unicode-numeric-superscripts-to-tex-ones
MathJax = {
    loader: {load: ["[tex]/ams", "[tex]/mathtools", "[tex]/physics"]},
    tex: {
        packages: {"[+]": ["ams", "mathtools", "physics"]},
        preFilters: [
            // Define pre-filter to convert Unicode superscripts to TeX syntax
            // NOTE: math is a MathItem object, math.math is a string.
            ({math}) => {
                math.math = unicodeLetterModifiersToTeX(math.math);
            },
            ({math}) => {
                math.math = unicodeSubscriptsToTeX(math.math);
            },
            ({math}) => {
                math.math = unicodeSuperscriptsToTeX(math.math);
            },
        ],
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        macros: {
            // operators
            argmax: "\\operatorname*{arg\\,max}",
            argmin: "\\operatorname*{arg\\,min}",
            Median: "\\operatorname{Median}",
            diag: "\\operatorname{diag}",
            dist: "\\operatorname{dist}",
            logsumexp: "\\operatorname{logsumexp}",
            NdtriExp: "\\operatorname{ndtri\\_exp}",
            rank: "\\operatorname{rank}",
            relu: "\\operatorname{ReLU}",
            KL: "\\operatorname{KL}",
            tr: "\\operatorname{tr}",
            E: "\\operatorname{\\mathbf{E}",
            Var: "\\operatorname{\\mathbf{Var}}",
            // macros
            bmat: ["\\begin{bmatrix} #1 \\end{bmatrix}", 1],
            norm: ["\\left\\lVert #1\\right\\rVert", 1],
            abs: ["\\left\\lvert #1\\right\\rvert", 1],
            set: ["\\left\\{ #1 \\right\\}", 1],
            seq: ["\\left( #1 \\right)", 1],
            tuple: ["\\left( #1 \\right)", 1],
            floor: ["\\left\\lfloor #1 \\right\\rfloor", 1],
            ceil: ["\\left\\lceil #1 \\right\\rceil", 1],
        },

    },
};
