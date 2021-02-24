TARGET_TEXTS_EN = [
    'Bleep-bloop, I am a robot!',
    'There is a number (12345) and an email (hello-there@box.com).'
]

TARGET_TEXTS_RU = [
    'Всем привет, я только проснулся)))',
    'Взрыв на федеральном газопроводе произошел в Оренбургской области'
]

COMPARISON_SETS_EN = [
    {'number', 'robot', 'bloop', 'email', 'bleep'},
    {'number', 'email', 'bleep'}
]

COMPARISON_SETS_RU = [
    {'взрыв', 'газопровод', 'область', 'оренбургский', 'привет', 'произойти',
     'проснуться', 'федеральный'},
    {}
]

# TODO: reliably compare topic scores?
COMPARISON_TOPICS_EN = [
    # Init test
    (
        91,
        {
            x for x, _ in
            [
                ('number', 0.21665008),
                ('bleep', 0.20454454),
                ('bloop', 0.20395425),
                ('robot', 0.19955926),
                ('email', 0.17529182)
            ]
        }
    ),

    # Update test
    (
        13,
        {
            x for x, _ in
            [
                ('bleep', 0.2),
                ('bloop', 0.2),
                ('robot', 0.2),
                ('email', 0.2),
                ('number', 0.2)
            ]
        }
    ),
    (
        82,
        {
            x for x, _ in
            [('bleep', 0.2),
             ('bloop', 0.2),
             ('robot', 0.2),
             ('email', 0.2),
             ('number', 0.2)]
        }
    )
]

COMPARISON_TOPICS_RU = [
    {}
]
