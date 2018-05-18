from errbot import BotPlugin, botcmd

class HelloBot(BotPlugin):
    """'Hello!' plugin for Errbot"""

    @botcmd
    def hello(self, msg, args):
        """Say hello to someone"""
        return "Hello, " + format(msg.frm)