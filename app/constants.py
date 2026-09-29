"""Immutable UI language labels, sample strings and translator instructions."""

LANGUAGES = {
    'en': 'English', 'de': 'Deutsch', 'fr': 'Français', 'es': 'Español',
    'it': 'Italiano', 'pt': 'Português', 'ru': 'Русский', 'he': 'עברית',
    'ar': 'العربية', 'ja': '日本語', 'zh': '中文',
}

MOCK_DICTIONARY = {
    'title': 'Welcome to our website', 'subtitle': 'Discover a simpler way to work',
    'home': 'Home', 'about': 'About us', 'contact': 'Contact',
    'get_started': 'Get started', 'learn_more': 'Learn more',
    'email_label': 'Email address', 'submit': 'Send message',
    'footer': 'All rights reserved.',
}

SYSTEM_PROMPT = (
    'You are a precise UI localization translator. Output valid JSON only. '
    'Any double quote that is part of translated text must be escaped as \\" inside the JSON string. '
    'Never output an unescaped double quote inside a JSON string.'
)

USER_PROMPT = (
    'Translate each string in the JSON array strictly into {language}. '
    'Use exactly the requested target language and do not substitute Russian, English, '
    'a related language, or another more common language. '
    'If you cannot reliably translate into exactly the requested target language, '
    'return the original input array unchanged. '
    'Return ONLY a JSON array of translated strings in EXACTLY the same order and count. '
    'Preserve placeholders like {{name}}, HTML tags, URLs, numbers and punctuation as appropriate. '
    'If translated text contains a double quote, escape it correctly for JSON as \\". '
    'Do not add explanations or markdown. Input: {values}'
)

CUSTOM_LANGUAGE_RECOGNITION_PROMPT = (
    'The user manually entered this target-language name: {language}. '
    'Identify whether it clearly refers to exactly one established language. '
    'The language may be natural or constructed. Accept established constructed languages '
    'when they have a recognizable name, documented grammar/vocabulary, and a standard language code. '
    'Examples include Esperanto, Klingon (tlhIngan Hol), and Quenya. '
    'The language name may itself be written in any language or script; identify the language it NAMES, '
    'not the language used to write the name. '
    'For example, Russian "норвежский" means Norwegian and Russian "Идиш" means Yiddish. '
    'Minor spelling mistakes are acceptable when the intended language is unambiguous. '
    'If the name is meaningless, invented by the user, ambiguous between multiple languages, '
    'or you cannot confidently identify one established language with a standard code, '
    'treat it as unrecognized. '
    'If recognized, return ONLY one valid JSON object in this exact shape: '
    '{{"language_recognized": true, "language": "<canonical English language name>", '
    '"language_code": "<standard BCP-47 or ISO language code>"}}. '
    'Use a canonical English language name for "language". '
    'Use the shortest standard language code that identifies the language unambiguously. '
    'If the language is not recognized, return exactly: '
    '{{"language_recognized": false, "language": null, "language_code": null}}. '
    'Do not translate any UI text and do not add explanations or markdown.'
)

JSON_RETRY_PROMPT = (
    '\nPrevious response was invalid JSON. Return the same translations, but ensure the JSON '
    'is valid and all internal double quotes are properly escaped.'
)

CUSTOM_LANGUAGE_RECOGNITION_JSON_RETRY_PROMPT = (
    '\nPrevious response was invalid JSON. Return the same language-recognition result, '
    'but ensure the JSON object is valid and contains only language_recognized, language, '
    'and language_code.'
)

FALLBACK_MESSAGE = 'Translation unavailable; original English strings returned'
UNSUPPORTED_LANGUAGE_MESSAGE = (
    'This language is recognized, but reliable translation is not currently supported'
)
LANGUAGE_NOT_RECOGNIZED_MESSAGE = 'Language not recognized'
