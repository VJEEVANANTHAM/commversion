import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chatbot import parse_budget, valid_email, valid_phone, valid_uk_postcode, LeadChatbot, M1, M2, M3
from data_access import PropertyData, normalize_postcode

def test_budget_formats():
    assert parse_budget("£1.5m") == 1500000
    assert parse_budget("1,200,000") == 1200000
    assert parse_budget("1.2 million") == 1200000
    assert parse_budget("950k") == 950000
    assert parse_budget("£999,999") == 999999

def test_validators():
    assert valid_email("a.person@example.co.uk")
    assert not valid_email("not-an-email")
    assert valid_phone("+44 7700 900123")
    assert not valid_phone("123")
    assert valid_uk_postcode("gu5 7aq")
    assert valid_uk_postcode("SW3 3RB")
    assert not valid_uk_postcode("not a postcode")

def test_postcode_normalization_and_seller_lookup():
    assert normalize_postcode("gu5 7aq") == "GU57AQ"
    data=PropertyData("data")
    assert data.is_seller_covered("GU1 3AD")

def test_new_build_budget_under_one_million_closes():
    bot=LeadChatbot("data")
    assert bot.reply("x","Jeevan") .startswith("Thanks")
    bot.reply("x","buy")
    bot.reply("x","no")
    bot.reply("x","new-build")
    answer=bot.reply("x","950k")
    assert "under £1 million" in answer
    assert bot.sessions["x"]["phone"] is None

def test_qualified_resale_flow_and_faq_side_question():
    bot=LeadChatbot("data")
    bot.reply("x","Sam")
    bot.reply("x","buy")
    bot.reply("x","no")
    bot.reply("x","resale")
    bot.reply("x","250k")
    # Pick a real postcode with an available resale listing.
    data=bot.data
    candidates=data.properties[(data.properties.category_norm=="resale") & (data.properties.status_norm=="available")]
    pc=candidates.iloc[0]["postcode"]
    answer=bot.reply("x",pc)
    assert "available matches" in answer
    assert bot.sessions["x"]["qualified"]
    # Side question must preserve the phone step.
    answer=bot.reply("x","What are your opening hours?")
    assert "phone number" in answer.lower()
    assert bot.sessions["x"]["step"]=="phone"

def test_seller_outside_coverage_does_not_collect_contact():
    bot=LeadChatbot("data")
    bot.reply("x","Alex")
    bot.reply("x","sell")
    answer=bot.reply("x","SW1A 1AA")
    assert "postcode you provided" in answer
    assert bot.sessions["x"]["phone"] is None
