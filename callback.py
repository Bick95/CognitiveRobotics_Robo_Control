import os

def callback(locals_, globals_):
    self_ = locals_['self']
    updates_ = locals_['update']

    if updates_ % self_.checkpoint_frequency == 0:
        if not os.path.exists(self_.check_point_location):
            os.makedirs(self_.check_point_location)

        check_point = updates_  # int(updates_ / self_.checkpoint_frequency)
        self_.save(self_.check_point_location + "checkpoint_" + str(check_point))

    return True
