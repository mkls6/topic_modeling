TARGET_TEXTS_EN = [
    'Bleep-bloop, I am a robot!',
    'There is a number (12345) and an email (hello-there@box.com).'
]

COMPARISON_SETS_EN = [
    {'numb', 'robot', 'bloop', 'email', 'bleep'},
    {'numb', 'email', 'bleep'}
]

# TODO: reliably compare topic scores?
COMPARISON_TOPICS_EN = [
    # Init test
    (
        91,
        {
            x for x, _ in
            [
                ('numb', 0.21665008),
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
                ('numb', 0.2)
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
             ('numb', 0.2)]
        }
    )
]
