class ChangeHandler:
    def handle_new_table(self, table):
        pass

    def handle_invalid_change(self, prev_table, current_table):
        pass

    def handle_valid_change(self, prev_table, current_table, gen, version, correspondences):
        pass

    def handle_table_done(self):
        pass

    def handle_complete(self):
        pass
