/****************************************************************************************************************
* Copyright: © 2018-2026 Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*) 
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file. 
* Authors: Ozan Nurettin Süel (aka UI-Manufaktur UG *R.I.P*)
*****************************************************************************************************************/
module uim.statistics.classes.dataset;

import uim.statistics;
@safe:

class Dataset : IValueset {
    private Sample[] _samples;

    this() @safe {}

    void add(Sample s) @safe {
        _samples ~= s;
    }

    size_t length() const @safe { return _samples.length; }

    double[] values() const @safe {
        double[] vals;
        vals.reserve(_samples.length);
        foreach (s; _samples) {
            vals ~= s.value;
        }
        return vals;
    }

    Sample[] samples() const @safe { return _samples.dup; }
}
