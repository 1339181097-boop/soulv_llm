from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_pipeline.data_utils import configure_console_output, ensure_parent_dir, log_info, log_success
from src.data_pipeline.handlers.handler_intent import DEFAULT_SYSTEM_PROMPT

DEFAULT_OUTPUT_PATH = "data/raw/intent.jsonl"
DEFAULT_TOTAL_SAMPLES = 800
DEFAULT_SEED = 42

INTENT_ORDER = [
    "FUNCTION_FLIGHTS_SEARCH_STRATEGY",
    "FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY",
    "FUNCTION_FLIGHTS_PASSENGER_STRATEGY",
    "FUNCTION_HOTELS_STRATEGY",
    "TRAVEL_STRATEGY",
    "TRAVEL_LOCATION_STRATEGY",
    "FUNCTION_TICKETS_STRATEGY",
    "FUNCTION_CAR_RENTAL_STRATEGY",
    "FUNCTION_VISA_STRATEGY",
    "DEFAULT_STRATEGY",
]

CITIES = [
    "هŒ—ن؛¬",
    "ن¸ٹوµ·",
    "ه¹؟ه·‍",
    "و·±هœ³",
    "و‌­ه·‍",
    "وˆگéƒ½",
    "é‡چه؛†",
    "è¥؟ه®‰",
    "ن¸‰ن؛ڑ",
    "وک†وکژ",
    "هژ¦é—¨",
    "هچ—ن؛¬",
    "è‹ڈه·‍",
    "é•؟و²™",
    "و­¦و±‰",
    "ه¤©و´¥",
    "é‌’ه²›",
    "ه¤§çگ†",
    "ن¸½و±ں",
    "ه“ˆه°”و»¨",
]

MONTHS = ["ن¸‹ه‘¨", "ن¸‹ن¸ھوœˆ", "ه›½ه؛†", "ن؛”ن¸€", "وڑ‘هپ‡", "وک¥èٹ‚", "وœ¬ه‘¨وœ«", "وکژه¤©", "هگژه¤©", "وœˆه؛•"]

AIRPORTS = [
    "é¦–éƒ½وœ؛هœ؛",
    "ه¤§ه…´وœ؛هœ؛",
    "وµ¦ن¸œوœ؛هœ؛",
    "è™¹و،¥وœ؛هœ؛",
    "ç™½ن؛‘وœ؛هœ؛",
    "ه®‌ه®‰وœ؛هœ؛",
    "ه¤©ه؛œوœ؛هœ؛",
    "هڈŒوµپوœ؛هœ؛",
    "ه‡¤ه‡°وœ؛هœ؛",
    "ه’¸éک³وœ؛هœ؛",
]

SPOTS = [
    "و•…ه®«",
    "è؟ھه£«ه°¼",
    "çژ¯çگƒه½±هںژ",
    "ه¤–و»©",
    "é”¦ç»£ن¸­هچژ",
    "é•؟éڑ†é‡ژç”ںهٹ¨ç‰©ه›­",
    "ه…µé©¬ن؟‘",
    "è¥؟و¹–",
    "çژ‰é¾™é›ھه±±",
    "ن¸œو–¹وکژçڈ ",
    "ه¤©ه‌›",
    "é¼“وµھه±؟",
]

COUNTRIES = [
    "و—¥وœ¬",
    "ç¾ژه›½",
    "è‹±ه›½",
    "و³•ه›½",
    "و„ڈه¤§هˆ©",
    "و–°هٹ ه‌،",
    "و³°ه›½",
    "éں©ه›½",
    "و¾³ه¤§هˆ©ن؛ڑ",
    "هٹ و‹؟ه¤§",
]

HOTEL_AREAS = [
    "وک¥ç†™è·¯",
    "è§£و”¾ç¢‘",
    "ه¤–و»©",
    "è¥؟و¹–",
    "ه¤©و²³è·¯",
    "ن¸‰é‡Œه±¯",
    "ن؛”ه››ه¹؟هœ؛",
    "هچ—é”£é¼“ه··",
    "é¼“و¥¼",
    "وµ·و£ و¹¾",
]

CAR_TYPES = ["SUV", "ه•†هٹ،è½¦", "7ه؛§è½¦", "ç»ڈوµژه‍‹è½؟è½¦", "و–°èƒ½و؛گè½¦"]
PASSENGER_FIELDS = ["وٹ¤ç…§هڈ·ç پ", "ن¹کوœ؛ن؛؛ن؟،وپ¯", "è؛«ن»½è¯پهڈ·", "è‹±و–‡ه§“هگچ", "ه¸¸ç”¨و—…ه®¢"]
PAYMENT_WORDS = ["çژ°هœ¨", "ç«‹هˆ»", "ن»ٹه¤©", "é©¬ن¸ٹ", "è؟™ن¼ڑه„؟"]
CUSTOMER_ROLES = ["وˆ‘", "وˆ‘ن»¬", "ن¸€ه®¶ن¸‰هڈ£", "ن¸¤ن¸ھن؛؛", "وˆ‘ه’Œوœ‹هڈ‹"]
DEFAULT_TOPICS = ["و´»هٹ¨", "ن¼کوƒ ", "ن¼ڑه‘ک", "ه®¢وœچ", "ه¹³هڈ°", "هٹںèƒ½", "وœچهٹ،èŒƒه›´", "ه”®هگژ"]


def _samples_per_intent(total_samples: int) -> dict[str, int]:
    base = total_samples // len(INTENT_ORDER)
    remainder = total_samples % len(INTENT_ORDER)
    counts: dict[str, int] = {}
    for index, name in enumerate(INTENT_ORDER):
        counts[name] = base + (1 if index < remainder else 0)
    return counts


def _flight_search_query(rng: random.Random) -> str:
    origin, destination = rng.sample(CITIES, 2)
    when = rng.choice(MONTHS)
    templates = [
        f"{when}{origin}é£‍{destination}çڑ„وœ؛ç¥¨ه¸®وˆ‘وں¥ن¸€ن¸‹",
        f"وˆ‘وƒ³çœ‹{origin}هˆ°{destination}çڑ„èˆھçڈ­",
        f"{when}ن»ژ{origin}هژ»{destination}è؟کوœ‰ç¥¨هگ—",
        f"ه¸®وˆ‘وگœن¸€ن¸‹{origin}é£‍{destination}وœ€ن¾؟ه®œçڑ„وœ؛ç¥¨",
        f"{origin}هˆ°{destination}çڑ„é£‍وœ؛ç¥¨çژ°هœ¨ه¤ڑه°‘é’±",
        f"وں¥ن¸‹{when}{origin}ه¾€è؟”{destination}èˆھçڈ­",
        f"وˆ‘ه‡†ه¤‡هژ»{destination}ï¼Œه…ˆçœ‹çœ‹ن»ژ{origin}ه‡؛هڈ‘çڑ„èˆھçڈ­",
        f"{origin}é£‍{destination}وکژو—©وœ€و—©ن¸€çڈ­èˆھçڈ­وک¯ن»€ن¹ˆ",
    ]
    return rng.choice(templates)


def _flight_confirm_query(rng: random.Random) -> str:
    destination = rng.choice(CITIES)
    when = rng.choice(MONTHS)
    actor = rng.choice(CUSTOMER_ROLES)
    pay_word = rng.choice(PAYMENT_WORDS)
    templates = [
        f"{actor}{when}هژ»{destination}è؟™çڈ­وœ؛ç¥¨çژ°هœ¨ن¸‹هچ•",
        f"ه¸®وˆ‘ç،®è®¤ن¸‹هچ•è؟™ه¼ هژ»{destination}çڑ„وœ؛ç¥¨",
        f"è؟™ن¸ھهژ»{destination}çڑ„ç¥¨هڈ¯ن»¥ç›´وژ¥و”¯ن»کهگ—",
        f"هژ»{destination}è؟™çڈ­وœ؛ç¥¨وˆ‘{pay_word}ه°±è®¢",
        f"{when}é£‍{destination}è؟™ن¸ھèˆھçڈ­ç،®è®¤é¢„è®¢ه¹¶ن»کو¬¾",
        f"وٹٹهˆڑو‰چé‚£ن¸ھهژ»{destination}çڑ„èˆھçڈ­ن¸‹هچ•هگ§",
        f"è؟™ن¸ھهژ»{destination}çڑ„èˆھçڈ­و²،é—®é¢کçڑ„è¯‌ç›´وژ¥ه¸®وˆ‘وڈگن؛¤",
        f"وˆ‘ه†³ه®ڑن¹°هژ»{destination}çڑ„è؟™ه¼ وœ؛ç¥¨ï¼Œن¸‹ن¸€و­¥و€ژن¹ˆو”¯ن»ک",
    ]
    return rng.choice(templates)


def _passenger_query(rng: random.Random) -> str:
    field = rng.choice(PASSENGER_FIELDS)
    destination = rng.choice(CITIES)
    actor = rng.choice(CUSTOMER_ROLES)
    templates = [
        f"و€ژن¹ˆو·»هٹ {field}",
        f"{actor}وƒ³ن؟®و”¹ن¹کوœ؛ن؛؛çڑ„{field}",
        f"هژ»{destination}çڑ„è®¢هچ•é‡Œه¸¸ç”¨و—…ه®¢هœ¨ه“ھé‡Œç»´وٹ¤",
        f"ن¹کوœ؛ن؛؛ن؟،وپ¯ه،«é”™ن؛†ï¼Œ{field}و€ژن¹ˆو”¹",
        f"هڈ¯ن»¥و–°ه¢‍ن¸€ن¸ھن¹کوœ؛ن؛؛ç»™{destination}è؟™è¶ںè،Œç¨‹ç”¨هگ—",
        f"وٹ¤ç…§è؟‡وœںن؛†ï¼Œ{field}و€ژن¹ˆو›´و–°",
        f"ه¸®وˆ‘çœ‹çœ‹هژ»{destination}è؟™هچ•çڑ„ن¹کوœ؛ن؛؛èµ„و–™هœ¨ه“ھç®،çگ†",
        f"وˆ‘وƒ³هˆ é™¤ن¸€ن¸ھن¹کوœ؛ن؛؛ه¹¶é‡چه،«{field}",
    ]
    return rng.choice(templates)


def _hotel_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    area = rng.choice(HOTEL_AREAS)
    when = rng.choice(MONTHS)
    templates = [
        f"{when}هژ»{city}وƒ³è®¢é…’ه؛—",
        f"{city}{area}é™„è؟‘وœ‰ن»€ن¹ˆن½ڈه®؟وژ¨èچگ",
        f"ه¸®وˆ‘و‰¾ن¸€ن¸‹{city}ن½ڈه¾—و–¹ن¾؟çڑ„é…’ه؛—",
        f"وƒ³çœ‹{city}و°‘ه®؟ï¼Œوœ‰و²،وœ‰و€§ن»·و¯”é«کçڑ„",
        f"{when}هژ»{city}ï¼Œه¸®وˆ‘çœ‹çœ‹ن؛²ه­گé…’ه؛—",
        f"{city}ه¸‚ن¸­ه؟ƒé…’ه؛—و€ژن¹ˆé€‰",
        f"{city}{area}é™„è؟‘ن½ڈه“ھé‡Œو¯”è¾ƒو–¹ن¾؟",
        f"وژ¨èچگن¸€ن¸‹{city}çڑ„و°‘ه®؟وˆ–è€…é…’ه؛—",
    ]
    return rng.choice(templates)


def _travel_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    days = rng.choice(["1ه¤©", "2ه¤©", "3ه¤©", "4ه¤©", "5ه¤©"])
    templates = [
        f"{city}وœ‰ن»€ن¹ˆه¥½çژ©çڑ„",
        f"{city}{days}و—…و¸¸و”»ç•¥",
        f"هژ»{city}çژ©و€ژن¹ˆه®‰وژ’و¯”è¾ƒه¥½",
        f"{city}ه€¼ه¾—وژ¨èچگçڑ„و™¯ç‚¹وœ‰ه“ھن؛›",
        f"{days}و—¶é—´هœ¨{city}و€ژن¹ˆçژ©",
        f"ç¬¬ن¸€و¬،هژ»{city}ï¼Œç»™ن¸ھو”»ç•¥",
        f"{city}è‡ھç”±è،Œوœ‰ن»€ن¹ˆوژ¨èچگ",
        f"وƒ³هژ»{city}و—…و¸¸ï¼Œه…ˆçœ‹çœ‹و”»ç•¥",
    ]
    return rng.choice(templates)


def _travel_location_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    spot = rng.choice(SPOTS)
    days = rng.choice(["هچٹه¤©", "1ه¤©", "2ه¤©"])
    templates = [
        f"{city}{spot}{days}و”»ç•¥",
        f"{city}{spot}é™„è؟‘çژ©ن¸€ه¤©و€ژن¹ˆه®‰وژ’",
        f"{spot}ه€¼ن¸چه€¼ه¾—هژ»ï¼Œوœ‰ن»€ن¹ˆوژ¨èچگ",
        f"{city}{spot}و€ژن¹ˆçژ©و¯”è¾ƒهگˆé€‚",
        f"وˆ‘هœ¨{city}ï¼Œوƒ³هژ»{spot}çœ‹çœ‹",
        f"{spot}ه‘¨è¾¹è؟کوœ‰ن»€ن¹ˆو™¯ç‚¹هڈ¯ن»¥ن¸€èµ·çژ©",
        f"{city}{spot}و¸¸çژ©è·¯ç؛؟و€ژن¹ˆوژ’",
        f"{spot}é™„è؟‘ه¥½çژ©ه¥½هگƒçڑ„وœ‰ه“ھن؛›",
    ]
    return rng.choice(templates)


def _ticket_query(rng: random.Random) -> str:
    spot = rng.choice(SPOTS)
    templates = [
        f"{spot}é—¨ç¥¨ه¤ڑه°‘é’±",
        f"ه¸®وˆ‘ن¹°{spot}é—¨ç¥¨",
        f"{spot}و™¯هŒ؛ç¥¨çژ°هœ¨è؟کوœ‰هگ—",
        f"{spot}éœ€è¦پوڈگه‰چé¢„ç؛¦é—¨ç¥¨هگ—",
        f"{spot}ن¹گه›­ç¥¨و€ژن¹ˆن¹°",
        f"وں¥ن¸€ن¸‹{spot}وˆگن؛؛ç¥¨ن»·و ¼",
        f"{spot}ن»ٹه¤©çڑ„é—¨ç¥¨èƒ½è®¢هگ—",
        f"وˆ‘وƒ³ن¹°ن¸¤ه¼ {spot}é—¨ç¥¨",
    ]
    return rng.choice(templates)


def _car_rental_query(rng: random.Random) -> str:
    city = rng.choice(CITIES)
    car_type = rng.choice(CAR_TYPES)
    templates = [
        f"{city}ç§ں{car_type}ه¤ڑه°‘é’±",
        f"ه¸®وˆ‘çœ‹çœ‹{city}وœ؛هœ؛ç§ںè½¦",
        f"{city}è‡ھé©¾ç§ںè½¦و€ژن¹ˆè®¢",
        f"{city}وœ‰و²،وœ‰و—¥ç§ںè½¦وœچهٹ،",
        f"هژ»{city}و—…و¸¸وƒ³ç§ںè½¦",
        f"{city}ç§ںè½¦ن»·و ¼ه¤§و¦‚ه¤ڑه°‘",
        f"وژ¨èچگن¸€ن¸‹{city}é‌ è°±çڑ„ç§ںè½¦و–¹و،ˆ",
        f"{city}وƒ³ç§ںن¸€è¾†{car_type}ه‡؛و¸¸",
    ]
    return rng.choice(templates)


def _visa_query(rng: random.Random) -> str:
    country = rng.choice(COUNTRIES)
    templates = [
        f"{country}ç­¾è¯پو€ژن¹ˆهٹ‍",
        f"{country}و—…و¸¸ç­¾è¯پéœ€è¦پن»€ن¹ˆو‌گو–™",
        f"{country}ç­¾è¯پهٹ‍çگ†وµپç¨‹وک¯ن»€ن¹ˆ",
        f"وˆ‘وƒ³هٹ‍{country}و—…و¸¸ç­¾è¯پ",
        f"{country}ç­¾è¯په¤ڑن¹…èƒ½ن¸‹و‌¥",
        f"{country}ç­¾è¯پçژ°هœ¨ه¥½هٹ‍هگ—",
        f"ç”³è¯·{country}ç­¾è¯پè¦په‡†ه¤‡ه“ھن؛›èµ„و–™",
        f"{country}è‡ھç”±è،Œç­¾è¯پو€ژن¹ˆه¼„",
    ]
    return rng.choice(templates)


def _default_query(rng: random.Random) -> str:
    topic = rng.choice(DEFAULT_TOPICS)
    city = rng.choice(CITIES)
    when = rng.choice(MONTHS)
    actor = rng.choice(CUSTOMER_ROLES)
    templates = [
        "ن½ وک¯è°پ",
        "ن½ èƒ½ه¸®وˆ‘هپڑن»€ن¹ˆ",
        "و€ژن¹ˆèپ”ç³»ن؛؛ه·¥ه®¢وœچ",
        "ن½ ن»¬ه¹³هڈ°é‌ è°±هگ—",
        f"وœ€è؟‘وœ‰ن»€ن¹ˆ{topic}هگ—",
        f"وˆ‘وœ‰ç‚¹و²،وƒ³ه¥½هژ»ه“ھçژ©ï¼Œه…ˆç»™وˆ‘ن»‹ç»چن¸‹ن½ èƒ½وڈگن¾›ن»€ن¹ˆوœچهٹ،",
        f"ن½ ه’Œهˆ«çڑ„و—…و¸¸هٹ©و‰‹وœ‰ن»€ن¹ˆهŒ؛هˆ«",
        "ن½ ن»¬وک¯هپڑن»€ن¹ˆçڑ„",
        f"ن½ èƒ½ن¸چèƒ½ه…ˆه‘ٹè¯‰وˆ‘{city}ه€¼ن¸چه€¼ه¾—هژ»ï¼Œن¸چç”¨وژ¨èچگه…·ن½“ن؛§ه“پ",
        f"tripAI ن¸»è¦پèƒ½ه¸®وˆ‘è§£ه†³ه“ھن؛›و—…و¸¸é—®é¢ک",
        f"وˆ‘çژ°هœ¨è؟کو²،ه†³ه®ڑهژ»ه“ھï¼Œه…ˆéڑڈن¾؟èپٹèپٹهڈ¯ن»¥هگ—",
        f"ن½ ن»¬ه¹³هڈ°وœ‰ن»€ن¹ˆن¼ڑه‘کوˆ–è€…{topic}",
        f"{actor}{when}وƒ³ه‡؛هژ»èµ°èµ°ï¼Œن½ ه…ˆن»‹ç»چن¸€ن¸‹è‡ھه·±",
        f"وˆ‘è؟کو²،ç،®ه®ڑç›®çڑ„هœ°ï¼Œه…ˆé—®ن¸‹ن½ éƒ½و”¯وŒپه“ھن؛›و—…و¸¸وœچهٹ،",
        f"{city}ه…ˆن¸چèپٹه…·ن½“ن؛§ه“پï¼Œوˆ‘وƒ³çں¥éپ“ن½ èƒ½ه¸®وˆ‘هپڑه“ھن؛›ن؛‹وƒ…",
        f"ه¦‚و‍œوˆ‘هڈھوک¯وƒ³ه’¨è¯¢{topic}ï¼Œن½ èƒ½ه¤„çگ†هگ—",
        f"ن½ ن»¬ه’Œو™®é€ڑو—…و¸¸ه¹³هڈ°ç›¸و¯”وœ‰ن»€ن¹ˆن¸چن¸€و ·",
        f"وˆ‘è؟کن¸چه‡†ه¤‡ن¸‹هچ•ï¼Œهڈھوƒ³ه…ˆن؛†è§£ن¸€ن¸‹ه¹³هڈ°èƒ½هٹ›",
        f"{when}وˆ‘هڈ¯èƒ½وƒ³ه‡؛è،Œï¼Œن½†çژ°هœ¨ه…ˆé—®é—®ن½ èƒ½هپڑن»€ن¹ˆ",
        f"ه…ˆهˆ«وژ¨èچگه…·ن½“ه†…ه®¹ï¼Œه‘ٹè¯‰وˆ‘ن½ ن¸»è¦پè´ںè´£ه“ھن؛›ن¸ڑهٹ،",
    ]
    return rng.choice(templates)


QUERY_BUILDERS = {
    "FUNCTION_FLIGHTS_SEARCH_STRATEGY": _flight_search_query,
    "FUNCTION_FLIGHTS_CONFIGHTING_STRATEGY": _flight_confirm_query,
    "FUNCTION_FLIGHTS_PASSENGER_STRATEGY": _passenger_query,
    "FUNCTION_HOTELS_STRATEGY": _hotel_query,
    "TRAVEL_STRATEGY": _travel_query,
    "TRAVEL_LOCATION_STRATEGY": _travel_location_query,
    "FUNCTION_TICKETS_STRATEGY": _ticket_query,
    "FUNCTION_CAR_RENTAL_STRATEGY": _car_rental_query,
    "FUNCTION_VISA_STRATEGY": _visa_query,
    "DEFAULT_STRATEGY": _default_query,
}


def _build_record(intention_name: str, rng: random.Random, seen_queries: set[str]) -> dict[str, str]:
    query_builder = QUERY_BUILDERS[intention_name]
    for _ in range(500):
        user_query = query_builder(rng)
        if user_query not in seen_queries:
            seen_queries.add(user_query)
            return {
                "user_query": user_query,
                "intentionName": intention_name,
                "source": "synthetic_seed",
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
            }
    raise RuntimeError(f"و— و³•ن¸؛و„ڈه›¾ {intention_name} ç”ںوˆگè¶³ه¤ںه¤ڑçڑ„ن¸چé‡چه¤چ queryم€‚")


def generate_intent_seed_dataset(total_samples: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    counts = _samples_per_intent(total_samples)
    seen_queries: set[str] = set()
    records: list[dict[str, str]] = []

    for intention_name in INTENT_ORDER:
        for _ in range(counts[intention_name]):
            records.append(_build_record(intention_name, rng, seen_queries))

    rng.shuffle(records)
    return records


def write_jsonl(records: list[dict[str, str]], output_path: str) -> Path:
    path = ensure_parent_dir(output_path)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ç”ںوˆگ tripAI intent ç§چه­گو•°وچ®م€‚")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="è¾“ه‡؛ JSONL è·¯ه¾„م€‚")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL_SAMPLES, help="و€»و ·وœ¬و•°م€‚")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="éڑڈوœ؛ç§چه­گم€‚")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_console_output()
    log_info(f"ه¼€ه§‹ç”ںوˆگ intent ç§چه­گو•°وچ®ï¼Œو€»é‡ڈ {args.total} و‌،م€‚")
    records = generate_intent_seed_dataset(args.total, args.seed)
    output_path = write_jsonl(records, args.output)
    log_success(f"intent هژںه§‹و•°وچ®ه·²ç”ںوˆگï¼Œه…± {len(records)} و‌،م€‚")
    log_info(f"è¾“ه‡؛و–‡ن»¶: {output_path}")


if __name__ == "__main__":
    main()


