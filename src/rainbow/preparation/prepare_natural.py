# +
import spacy
#nlp = spacy.load("en")

def fact_to_prompt(kg, fact, attach_to="head"):
    head = fact["input_text"]
    relation = fact["prefix"]
    tail = fact["target_text"]

    if kg == "conceptnet" or kg == "transomcs":
        if relation == "AtLocation":
            prompt = "You are likely to find {} {} in {} ".format(
                article(head), head, article(tail)
            )
        elif relation == "CapableOf":
            prompt = "{} can ".format(head)
        elif relation == "CausesDesire":
            prompt = "{} would make you want to ".format(head)
        elif relation == "Causes":
            prompt = "Sometimes {} causes ".format(head)
        elif relation == "CreatedBy":
            prompt = "{} is created by".format(head)
        elif relation == "Desires":
            prompt = "{} {} desires".format(article(head), head)
        elif relation == "HasA":
            prompt = "{} {} ".format(head, posessive(head))
        elif relation == "HasPrerequisite":
            prompt = "{} requires ".format(vp_present_participle(head))
        elif relation == "HasProperty":
            prompt = "{} is ".format(head)
        elif relation == "MotivatedByGoal":
            prompt = "You would {} because you are ".format(head)
        elif relation == "ReceivesAction":
            prompt = "{} can be ".format(head)
        elif relation == "UsedFor":
            prompt = "{} {} is for ".format(article(head).upper(), head)
        elif (
            relation == "HasFirstSubevent"
            or relation == "HasSubevent"
            or relation == "HasLastSubevent"
        ):
            prompt = "While {}, you would ".format(vp_present_participle(head))
        elif relation == "InheritsFrom":
            prompt = "{} inherits from".format(head)
        elif relation == "PartOf":
            prompt = "{} {} is a part of {} ".format(
                article(head).upper(), head, article(tail)
            )
        elif relation == "IsA":
            prompt = "{} is {} ".format(head, article(tail))
        elif relation == "InstanceOf":
            prompt = "{} is an instance of".format(head)
        elif relation == "MadeOf":
            prompt = "{} is made of".format(head)
        elif relation == "DefinedAs":
            prompt = "{} is defined as ".format(head)
        elif relation == "NotCapableOf":
            prompt = "{} is not capable of".format(head)
        elif relation == "NotDesires":
            prompt = "{} {} does not desire".format(article(head), head)
        elif relation == "NotHasA":
            prompt = "{} does not have a".format(head)
        elif relation == "NotHasProperty" or relation == "NotIsA":
            prompt = "{} is not".format(head)
        elif relation == "NotMadeOf":
            prompt = "{} is not made of".format(head)
        elif relation == "SymbolOf":
            prompt = "{} is a symbol of".format(head)
        else:
            raise Exception(relation)
    elif kg == "atomic" or kg == "atomic2020":
        if attach_to == "head":
            if relation == "AtLocation":
                prompt = "You are likely to find {} {} in {} ".format(
                    article(head), head, article(tail)
                )
            elif relation == "CapableOf":
                prompt = "{} can ".format(head)
            elif relation == "Causes":
                prompt = "Sometimes {} causes ".format(head)
            elif relation == "Desires":
                prompt = "{} {} desires".format(article(head), head)
            elif relation == "HasProperty":
                prompt = "{} is ".format(head)
            elif relation == "HasSubEvent":
                prompt = "While {}, you would ".format(vp_present_participle(head))
            elif relation == "HinderedBy":
                prompt = "{}. This would not happen if"
            elif relation == "MadeUpOf":
                prompt = "{} {} contains".format(article(head), head)
            elif relation == "NotDesires":
                prompt = "{} {} does not desire".format(article(head), head)
            elif relation == "ObjectUse":
                prompt = "{} {} can be used for".format(article(head), head)
            elif relation == "isAfter":
                prompt = "{}. Before that, ".format(head)
            elif relation == "isBefore":
                prompt = "{}. After that, ".format(head)
            elif relation == "isFilledBy":
                prompt = "{} is filled by".format(head)  # TODO
            elif relation == "oEffect":
                prompt = "{}. The effect on others will be".format(head)
            elif relation == "oReact":
                prompt = "{}. As a result, others feel".format(head)
            elif relation == "oWant":
                prompt = "{}. After, others will want to".format(head)
            elif relation == "xAttr":
                prompt = "{}. PersonX is".format(head)
            elif relation == "xEffect":
                prompt = "{}. The effect on PersonX will be".format(head)
            elif relation == "xIntent":
                prompt = "{}. PersonX did this to".format(head)
            elif relation == "xNeed":
                prompt = "{}. Before, PersonX needs to".format(head)
            elif relation == "xReact":
                prompt = "{}. PersonX will be".format(head)
            elif relation == "xReason":
                prompt = "{}. PersonX did this because".format(head)
            elif relation == "xWant":
                prompt = "{}. After, PersonX will want to".format(head)
        else:
            if relation == "oEffect":
                prompt = "The effect on others will be {}".format(tail)
            elif relation == "oReact":
                prompt = "As a result, others feel {}".format(tail)
            elif relation == "oWant":
                prompt = "After, others will want to {}".format(tail)
            elif relation == "xAttr":
                prompt = "PersonX is {}".format(tail)
            elif relation == "xEffect":
                prompt = "The effect on PersonX will be {}".format(tail)
            elif relation == "xIntent":
                prompt = "PersonX did this to {}".format(tail)
            elif relation == "xNeed":
                prompt = "Before, PersonX needs to {}".format(tail)
            elif relation == "xReact":
                prompt = "PersonX will be {}".format(tail)
            elif relation == "xReason":
                prompt = "PersonX did this because {}".format(tail)
            elif relation == "xWant":
                prompt = "After, PersonX will want to {}".format(tail)
    else:
        raise Exception("Invalid KG")

    return prompt.strip()


def article(word):
    return "an" if word[0] in ["a", "e", "i", "o", "u"] else "a"


def posessive(word):
    if inflection_engine.singular_noun(word) is False:
        return "have"
    else:
        return "has"


import inflect

inflection_engine = inflect.engine()


def vp_present_participle(phrase):
    return phrase
    doc = phrase
    return " ".join(
        [
            inflection_engine.present_participle(token.text)
            if token.pos_ == "VERB" and token.tag_ != "VGG"
            else token.text
            for token in doc
        ]
    )


def read_jsonl_lines(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


