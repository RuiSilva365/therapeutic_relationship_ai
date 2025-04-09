from datetime import datetime, timedelta

def filter_last_week_messages(messages, reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()

    one_week_ago = reference_date - timedelta(days=7)
    filtered = [
        msg for msg in messages
        if datetime.fromtimestamp(msg["timestamp_ms"] / 1000) >= one_week_ago
    ]
    return filtered
