import modal

app = modal.App()


@app.function()
async def main():
    # TESTING CODE, REMOVE
    import os

    print("Ok, it ran")

    print("contents of /remote:")
    print(os.listdir("/remote"))

    # END TESTING CODE


@app.local_entrypoint()
def main_local():
    print("Running main locally")
    main.remote()
