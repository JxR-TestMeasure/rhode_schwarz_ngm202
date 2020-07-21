from scpi_instrument_common import Validate


class ValidateChannel(Validate):
    def __init__(self):
        Validate().__init__()

    def voltage(self, value):
        voltage_values = (0.0, 20.05), ('min', 'max')
        return self.float_rng_and_str_tuples(voltage_values, value, 3)

    def current(self, value):
        current_values = (0.001, 6.0), ('min', 'max')
        return self.float_rng_and_str_tuples(current_values, value, 4)

    def impedance(self, value):
        impedance_values = (-0.05, 100.0), ('min', 'max')
        return self.float_rng_and_str_tuples(impedance_values, value, 3)

    def resistance(self, value):
        resistance_values = (0.0, 10000.0), ('min', 'max')
        return self.float_rng_and_str_tuples(resistance_values, value, 3)

    def current_range(self, value):
        current_range_values = (10.0, 1.0, 0.1, 0.01), ('auto', 'min', 'max')
        return self.float_and_str_tuples(current_range_values, value)

    def voltage_range(self, value):
        voltage_range_values = (20.0, 5.0), 'auto'
        return self.float_and_str_tuples(voltage_range_values, value)

    def mode(self, value):
        mode_values = ('AUTO', 'SOURce', 'SINK', 'SOUR')
        return self.str_tuple(mode_values, value)

    def on_off(self, value):
        on_off_values = (0, 1), ('on', 'off')
        return self.int_rng_and_str_tuples(on_off_values, value)

    def output_delay_duration(self, value):
        output_delay_duration_values = (0.001, 10.0), ('on', 'off')
        return self.float_rng_and_str_tuples(output_delay_duration_values, value, 3)

    def channel(self, value):
        channel_values = (1, 2), ('1', '2')
        return self.int_rng_and_str_tuples(channel_values, value)

    def ramp_duration(self, value):
        ramp_duration_values = (0.01, 10.0), ('min', 'max', 'DEFault', 'DEF')
        return self.float_rng_and_str_tuples(ramp_duration_values, value, 2)

    def trigger_behavior(self, value):
        trigger_behavior_values = ('ON', 'OFF', 'GATed', 'GAT')
        return self.str_tuple(trigger_behavior_values, value)


class ValidateArbitrary(ValidateChannel):
    def __init__(self):
        ValidateChannel.__init__(self)

    def voltage(self, value):
        voltage_values = (0.0, 20.0)
        return self.float_rng_tuple(voltage_values, value, 3)

    def current(self, value):
        current_values = (0.001, 6.0)
        return self.float_rng_tuple(current_values, value, 4)

    def dwell_time(self, value):
        dwell_time_values = (0.001, 1728000.0)
        return self.float_rng_tuple(dwell_time_values, value, 3)

    def interpolation(self, value):
        interpolation_values = (0, 1)
        return self.int_rng_tuple(interpolation_values, value)

    def repetition(self, value):
        repetition_values = (0, 65535)
        return self.int_rng_tuple(repetition_values, value)

    def end_behavior(self, value):
        repetition_values = ('off', 'hold')
        return self.str_tuple(repetition_values, value)

    def arb_list(self, arb_list: dict):
        errors = 0
        for x in arb_list.keys():
            val = []
            val.append(self.voltage(arb_list[x]['voltage']))
            val.append(self.current(arb_list[x]['current']))
            val.append(self.dwell_time(arb_list[x]['dwell_time']))
            val.append(self.interpolation(arb_list[x]['interpolation']))
            for y in val:
                if isinstance(y, (ValueError, TypeError)):
                    print('Point:' + str(x))
                    print(self.error_text('WARNING', y))
                    errors += 1
        if errors == 0:
            return True
        else:
            print('Error count: ' + str(errors))
            return False

    def trigger_mode(self, value):
        trigger_mode_values = ('SINGle', 'SING', 'RUN')
        return self.str_tuple(trigger_mode_values, value)


class ValidateLog(Validate):
    def __init__(self):
        Validate.__init__(self)

    def count(self, value):
        count_values = (1, 10000000), ('min', 'max')
        return self.int_rng_and_str_tuples(count_values, value)

    def duration(self, value):
        duration_values = (1, 349000), ('min', 'max')
        return self.int_rng_and_str_tuples(duration_values, value)

    def interval(self, value):
        interval_values = (0.1, 600.0), ('min', 'max')
        return self.float_rng_and_str_tuples(interval_values, value, 1)

    def mode(self, value):
        mode_values = ('UNLimited', 'COUNt', 'DURation', 'SPAN', 'COUN', 'UNL', 'DUR')
        return self.str_tuple(mode_values, value)

    def on_off(self, value):
        on_off_values = (0, 1), ('on', 'off')
        return self.int_rng_and_str_tuples(on_off_values, value)


class ValidateDisplay(Validate):
    def __init__(self):
        Validate().__init__()

    def brightness(self, value):
        channel_values = (0.0, 1.0), ('min', 'max')
        return self.float_rng_and_str_tuples(channel_values, value, 1)


class ValidateFastLog(ValidateChannel):
    def __init__(self):
        ValidateChannel.__init__(self)

    def sample_rates(self, value):
        sample_rates_values = ('S15', 'S30', 'S61',
                               'S122', 'S244', 'S488',
                               'S976', 'S1K953', 'S3K906',
                               'S7K812', 'S15K625', 'S31K25',
                               'S62K5', 'S125K', 'S250K',
                               'S500K'
                               )
        return self.str_tuple(sample_rates_values, value)

    def write_duration(self, value):
        write_duration_values = (0.1, 99999.9)
        return self.float_rng_tuple(write_duration_values, value, 1)


class ValidateTrigger(Validate):
    def __init__(self):
        Validate().__init__()

    def state(self, value):
        state_values = (0, 1)
        return self.int_rng_tuple(state_values, value)

    def source(self, value):
        source_values = ('OUTPut', 'OMODe', 'DIO', 'OUTP', 'OMOD')
        return self.str_tuple(source_values, value)

    def channel(self, value):
        channel_values = (1, 2), ('1', 'OUT1', 'OUTP1', 'OUTPut1',
                                  '2', 'OUT2', 'OUTP2', 'OUTPut2')
        return self.int_rng_and_str_tuples(channel_values, value)

    def pin(self, value):
        pin_values = ('IN', 'EXT')
        return self.str_tuple(pin_values, value)

    def output_mode(self, value):
        output_mode_values = ('CC', 'CV', 'CR', 'SINK', 'PROTection', 'PROT')
        return self.str_tuple(output_mode_values, value)


class ValidateProtection(ValidateChannel):
    def __init__(self):
        ValidateChannel.__init__(self)

    def delay_initial(self, value):
        delay_initial_values = (0.0, 60.0), ('min', 'max')
        return self.float_rng_and_str_tuples(delay_initial_values, value, 2)

    def delay(self, value):
        delay_values = (0.0, 10.0), ('min', 'max')
        return self.float_rng_and_str_tuples(delay_values, value, 2)

    def link(self, value):
        link_values = (1, 2)
        return self.int_rng_tuple(link_values, value)

    def power(self, value):
        delay_values = (0.0, 60.0), ('min', 'max')
        return self.float_rng_and_str_tuples(delay_values, value, 3)






