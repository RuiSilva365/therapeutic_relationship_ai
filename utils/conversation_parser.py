def create_interaction_blocks(messages: list):
    blocks = []
    for i in range(len(messages) - 1):
        input_msg = messages[i]
        response_msg = messages[i + 1]

        # Só criar pares quando forem de pessoas diferentes
        if input_msg["sender_name"] != response_msg["sender_name"]:
            blocks.append({
                "input": {
                    "sender": input_msg["sender_name"],
                    "message": input_msg.get("content", "")
                },
                "response": {
                    "sender": response_msg["sender_name"],
                    "message": response_msg.get("content", "")
                }
            })
    return blocks
