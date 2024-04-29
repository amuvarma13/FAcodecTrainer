punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# chinese
zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
num_zh_tones = 6

# japanese
ja_symbols = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
num_ja_tones = 2

# English
en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
num_en_tones = 4

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}

# Chinese to IPA mapping
zh_to_ipa = {
    "E": "ə",
    "En": "ən",
    "a": "a",
    "ai": "ai",
    "an": "an",
    "ang": "aŋ",
    "ao": "aʊ",
    "b": "b",
    "c": "tsʰ",
    "ch": "ʈʂʰ",
    "d": "d",
    "e": "ɛ",
    "ei": "ei",
    "en": "ən",
    "eng": "əŋ",
    "er": "ɚ",
    "f": "f",
    "g": "g",
    "h": "x",
    "i": "i",
    "i0": "i",
    "ia": "ia",
    "ian": "ian",
    "iang": "iaŋ",
    "iao": "iaʊ",
    "ie": "ie",
    "in": "in",
    "ing": "iŋ",
    "iong": "ioŋ",
    "ir": "ɚ",
    "iu": "iou",
    "j": "tɕ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ong": "oŋ",
    "ou": "ou",
    "p": "p",
    "q": "tɕh",
    "r": "ɻ",
    "s": "s",
    "sh": "ʂ",
    "t": "t",
    "u": "u",
    "ua": "ua",
    "uai": "uai",
    "uan": "uan",
    "uang": "uaŋ",
    "ui": "uei",
    "un": "uen",
    "uo": "uo",
    "v": "v",
    "van": "v-an",
    "ve": "v-e",
    "vn": "v-n",
    "w": "w",
    "x": "ɕ",
    "y": "i",
    "z": "ts",
    "zh": "ʈʂ",
    "AA": "a",
    "EE": "i",
    "OO": "o"
}

# Japanese to IPA mapping
ja_to_ipa = {
    "N": "ɴ",
    "a": "a",
    "a:": "aː",
    "b": "b",
    "by": "bʲ",
    "ch": "tɕ",
    "d": "d",
    "dy": "dʲ",
    "e": "e",
    "e:": "eː",
    "f": "ɸ",
    "g": "ɡ",
    "gy": "ɡʲ",
    "h": "h",
    "hy": "ç",
    "i": "i",
    "i:": "iː",
    "j": "dʑ",
    "k": "k",
    "ky": "kʲ",
    "m": "m",
    "my": "mʲ",
    "n": "n",
    "ny": "ɲ",
    "o": "o",
    "o:": "oː",
    "p": "p",
    "py": "pʲ",
    "q": "kʷ",
    "r": "ɾ",
    "ry": "ɾʲ",
    "s": "s",
    "sh": "ɕ",
    "t": "t",
    "ts": "ts",
    "ty": "tʲ",
    "u": "u",
    "u:": "uː",
    "w": "ɰ",
    "y": "j",
    "z": "z",
    "zy": "zʲ"
}

# English to IPA mapping
en_to_ipa = {
    "aa": "ɑ",
    "ae": "æ",
    "ah": "ʌ",
    "ao": "ɔ",
    "aw": "aʊ",
    "ay": "aɪ",
    "b": "b",
    "ch": "ʧ",
    "d": "d",
    "dh": "ð",
    "eh": "ɛ",
    "er": "ɜːr",
    "ey": "eɪ",
    "f": "f",
    "g": "ɡ",
    "hh": "h",
    "ih": "ɪ",
    "iy": "iː",
    "jh": "ʤ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "ng": "ŋ",
    "ow": "oʊ",
    "oy": "ɔɪ",
    "p": "p",
    "r": "ɹ",
    "s": "s",
    "sh": "ʃ",
    "t": "t",
    "th": "θ",
    "uh": "ʊ",
    "uw": "uː",
    "V": "ə",
    "w": "w",
    "y": "j",
    "z": "z",
    "zh": "ʒ"
}

ipa_mapping = {
    **zh_to_ipa,
    **ja_to_ipa,
    **en_to_ipa
}


if __name__ == "__main__":
    a = set(zh_symbols)
    b = set(en_symbols)
    print(sorted(a & b))