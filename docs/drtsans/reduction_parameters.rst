====================
Reduction Parameters
====================

.. contents::

Default Values
--------------

.. exec::
    from drtsans.redparms import _instrument_json_generator
    # render the default values of the reduction parameter for each instrument suitable to restructured text
    docs = [default_json.to_rest() for _, default_json in _instrument_json_generator()]
    print(r'{}'.format(''.join(docs)))  # we require a raw string

Descriptions
------------

.. exec::
    from drtsans.redparms import _instrument_json_generator
    # render the default values of the reduction parameter for each instrument suitable to restructured text
    docs = ''
    for name, default_json in _instrument_json_generator(field='description'):
        default_json._json['instrumentName'] = name
        docs += default_json.to_rest()
    print(r'{}'.format(docs))  # we require a raw string
