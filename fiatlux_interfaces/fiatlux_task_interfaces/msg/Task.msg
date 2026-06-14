# A unique ID for the bulb replacement task.
string id

# The type of bulb which should be replaced, e.g., "led", "incandescent".
string bulb_type

# The name of the bulb.
string bulb_name

# The type of socket into which the bulb should be inserted, e.g., "e26".
string socket_type

# The name of the socket.
string socket_name

# The name of the lamp fixture on which the socket exists, e.g., "desk_lamp_0".
string target_fixture_name

# The number of seconds from the moment this request is received by when the task must complete.
uint64 time_limit
