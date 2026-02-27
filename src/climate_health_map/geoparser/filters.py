import re

TEXT_FILTER = re.compile(
    # copyright boilerplate
    r'Copyright \(C\)[^©]*|'
    r'\([C-c]\) [1-2][0-9]{3} Elsevier|'
    r'Published by Elsevier|'
    r'\. \(C\) [1-2][0-9]{3} |'
    r'\. \(C\) Copyright|'
    # International climate agreements:
    # Berlin
    r'(berlin(?:\S* ){0,15}cop)|(cop(?:\S* ){0,15}berlin)|'
    # Cancun
    r'(cancun(?:\S* ){0,15}cop)|(cop(?:\S* ){0,15}cancun)|'
    r'cancun pledge|'
    # Copenhagen
    r'(copenhagen(?:\S* ){0,15}cop)|(cop(?:\S* ){0,15}copenhagen)|'
    r'(copenhagen(?:\S* ){0,3}accord)|(accord(?:\S* ){0,3}copenhagen)|'
    # Glasgow
    r'(glasgow(?:\S* ){0,15}cop)|(cop(?:\S* ){0,15}glasgow)|'
    # Kyoto
    r'kyoto agreement|'
    r'kyoto commitments?|'
    r'kyoto emission|'
    r'kyoto framework|'
    r'kyoto gas|'
    r'kyoto process|'
    r'kyoto protocol'
    r'kyoto target|'
    # London
    r'london protocol|'
    # Montreal
    r'montreal protocol|'
    r'(montreal(?:\S* ){0,15}cop)|(cop(?:\S* ){0,15}montreal)|'
    # New Dehli
    r'(framework convention on climate change(?:\S* ){0,15}new delhi)|(new delhi(?:\S* ){0,15}framework convention on climate change)|'
    r'(unfccc(?:\S* ){0,15}new delhi)|(new delhi(?:\S* ){0,15}unfccc)'
    # Paris
    r'paris agreement|'
    r'(paris(?:\S* ){0,15}agreement)|(cop(?:\S* ){0,15}agreement)|'
    r'(paris(?:\S* ){0,15}cop)|(cop(?:\S* ){0,15}paris)',
    flags=re.I,
)  # ignore case
