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
			if (!buffer) { parts.push(s.slice(start, i)); }
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
		const mapped = SUPERSCRIPT_MAP.get(ch);

		if (mapped !== undefined) {  // starting or continuing superscript
			if (!buffer) { parts.push(s.slice(start, i)); }
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


// SEE: https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax
MathJax = {
	loader: { load: ["[tex]/ams", "[tex]/mathtools", "[tex]/physics"] },
	tex: {
		packages: {"[+]": ["ams", "mathtools", "physics"]},
		preFilters: [
		    // Define pre-filter to convert Unicode superscripts to TeX syntax
            // SEE: https://docs.mathjax.org/en/v4.0/advanced/synchronize/filters.html#converting-unicode-numeric-superscripts-to-tex-ones
			// NOTE: math is a MathItem object, math.math is a string.
			({ math }) => { math.math = unicodeSuperscriptsToTeX(math.math); },
			({ math }) => { math.math = unicodeSubscriptsToTeX(math.math); },
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
