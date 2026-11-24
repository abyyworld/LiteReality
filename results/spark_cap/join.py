def join(page):
    """Get past the join card: a name, webcam off, Join."""
    page.get_by_placeholder("e.g. Alex").fill("bench")
    box = page.get_by_label("Show my webcam on my avatar")
    if box.is_checked():
        box.uncheck()
    page.get_by_role("button", name="Join the room").click()
    page.wait_for_timeout(4000)
