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
    'Translate each string in the JSON array into {language}. '
    'Return ONLY a JSON array of translated strings in EXACTLY the same order and count. '
    'Preserve placeholders like {{name}}, HTML tags, URLs, numbers and punctuation as appropriate. '
    'If translated text contains a double quote, escape it correctly for JSON as \\". '
    'Do not add explanations or markdown. Input: {values}'
)

CUSTOM_LANGUAGE_USER_PROMPT = (
    'The user manually entered this target-language name: {language}. '
    'First decide whether it clearly identifies one real human language. '
    'Minor spelling mistakes are acceptable when the intended language is unambiguous. '
    'If the name is fictional, meaningless, ambiguous between multiple languages, or you cannot '
    'confidently identify one real language, treat it as unrecognized. '
    'If recognized, translate each string in the JSON array into that language. '
    'Return ONLY one valid JSON object in this exact shape: '
    '{{"language_recognized": true, "language": "<normalized language name>", '
    '"translations": ["...", "..."]}}. '
    'Use the standard self-name/native name for "language" when possible. '
    'The translations array must contain EXACTLY the same number of strings and preserve the input order. '
    'Preserve placeholders like {{name}}, HTML tags, URLs, numbers and punctuation as appropriate. '
    'If translated text contains a double quote, escape it correctly for JSON as \\". '
    'If the language is not recognized, return exactly: '
    '{{"language_recognized": false, "language": null, "translations": []}}. '
    'Do not invent a language and do not add explanations or markdown. Input: {values}'
)

JSON_RETRY_PROMPT = (
    '\nPrevious response was invalid JSON. Return the same translations, but ensure the JSON '
    'is valid and all internal double quotes are properly escaped.'
)

CUSTOM_LANGUAGE_JSON_RETRY_PROMPT = (
    '\nPrevious response was invalid JSON. Return the same language-recognition result and translations, '
    'but ensure the JSON is valid and all internal double quotes are properly escaped.'
)

FALLBACK_MESSAGE = 'Translation unavailable; original English strings returned'
LANGUAGE_NOT_RECOGNIZED_MESSAGE = 'Language not recognized'
