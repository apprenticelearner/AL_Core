# Copyright 2014-2015 Carnegie Mellon University

# TODO document this module

 # Don't edit the __version__ here, edit the VERSION file and use the setversions script
__version__ = '2.0.1'


# First a bunch of code to gracefully tell us if the Python we're trying
# to run in is too old.

from sys import version_info, exit

MINIMUM_PYTHON_VERSION = (3, 2)

try:
    if version_info < MINIMUM_PYTHON_VERSION:
        exit("PyIBL requires version {} or higher, but it appears version {} is running.".format(
            ".".join(map(str, MINIMUM_PYTHON_VERSION)), ".".join(map(str, version_info[0:3]))))
except:
    exit("PyIBL cannot run in this version of Python")

# The above won't even get a chance to run and exit gracefully if Python
# thinks there's a syntax error, so when it does, at least ensure the
# context it spits out points the poor user in the right direction.
if False: # We never want to actually run this code
    # This is Python 3 print syntax, which causes an otherwise inscrutable error in Python 2
    print("If Python complains of a syntax error here, it is too old a version to run PyIBL.", file=None)


# Now we're convinced we should be able to run in the currently executing
# version of Python.

from sys import hash_info, stdout
from warnings import warn
from verlib import NormalizedVersion
from collections import OrderedDict, Counter
from itertools import chain, repeat, count
from numbers import Number
from ordered_set import OrderedSet
from prettytable import PrettyTable, from_csv
import math
import random
import csv
import sqlite3

DEFAULT_NOISE = 0
ZERO_NOISE_TOLERANCE = 0.0001
ZERO_NOISE_DEFAULT_TEMPERATURE = 1
MINIMUM_TEMPERATURE = 0.1
DEFAULT_DECAY = 0
DEFAULT_ACTIVATION = float("-inf")

MINIMUM_PRECISION = 1
MAXIMUM_PRECISION = 10
DEFAULT_PRECISION = 4

LOG_HEADING_DEFAULT = True
DEFAULT_CSV_DIALECT = 'excel'

CREATE_DB_DEFAULT = False
CLEAR_DB_DEFAULT = False

ACTIVATION_CACHE_INCREMENT = 100
DB_BUFFER_LENGTH = 300

# TODO Ensure appropriate error behavior for bad arguments to public methods.
# TODO Unit tests!


# Logging setup

LOG_NONE = 0
LOG_TRIALS = 1
LOG_OPTIONS  = 2
LOG_INSTANCES  = 3

logDetails = OrderedSet()
logLevels = {}
logTypes = {}
logIndecies = {}
logCount = 0

def defineLogDetail(name, level, typ=None):
    global logCount
    logDetails.add(name)
    logLevels[name] = level
    if typ:
        logTypes[name] = typ
        logIndecies[name] = logCount
        logCount += 1

defineLogDetail("unusedDecisions", LOG_NONE)
defineLogDetail("unusedSituations", LOG_NONE)
# The order the following are defined in is also the order of columns in the log files.
defineLogDetail("sequence", LOG_NONE, "INTEGER")
defineLogDetail("tAgent", LOG_TRIALS, "TEXT")
defineLogDetail("tDecayParam", LOG_TRIALS, "REAL")
defineLogDetail("tNoiseParam", LOG_TRIALS, "REAL")
defineLogDetail("tTemperature", LOG_TRIALS, "REAL")
defineLogDetail("tBlock", LOG_TRIALS, "TEXT")
defineLogDetail("tTrial", LOG_TRIALS, "INTEGER")
defineLogDetail("tIteration", LOG_TRIALS, "INTEGER")
defineLogDetail("tChoice", LOG_TRIALS, "TEXT")
defineLogDetail("tChoiceSituation", LOG_TRIALS, "TEXT")
defineLogDetail("tResponse", LOG_TRIALS, "REAL")
defineLogDetail("oDecision", LOG_OPTIONS, "TEXT")
defineLogDetail("oSituation", LOG_OPTIONS, "TEXT")
defineLogDetail("oBlendedValue", LOG_OPTIONS, "REAL")
defineLogDetail("iUtility", LOG_INSTANCES, "REAL")
defineLogDetail("iReasonAdded", LOG_INSTANCES, "TEXT")
defineLogDetail("iFrequency", LOG_INSTANCES, "INTEGER")
defineLogDetail("iOccurrences", LOG_INSTANCES, "TEXT")
defineLogDetail("iActivationBase", LOG_INSTANCES, "REAL")
defineLogDetail("iActivationNoise", LOG_INSTANCES, "REAL")
defineLogDetail("iActivation", LOG_INSTANCES, "REAL")
defineLogDetail("iRetrievalProbability", LOG_INSTANCES, "REAL")

LOG_ALL = frozenset(logDetails)
"""A :class:`frozenset` containing all the possible option strings controlling logging detail."""

class _Unique:

    def __init__(self, name):
        self._repr = "<pyibl.{}>".format(name)

    def __repr__(self):
        return self._repr

NEXT = _Unique("NEXT")
"""A constant used for incrementing the value of :attr:`Population.block` property of a :class:`Population`.
When that property is set to ``NEXT``, if it is already a number, it will be incremented
by 1; otherwise it will be set to 1.
"""


def _safeFormat(value, fmt):
    try:
        return format(value, fmt)
    except:
        return value


class SituationDecision:
    """A possible decision paired with a collection current values of attributes, for an :class:`Agent` to choose between.
    A SituationDecions should only be created by calling an agent's
    :meth:`Agent.situationDecision` method, which will ensure it has the correct
    attributes for that agent. The attribute names are strings. The decision and all
    attribute values must be hashable Python objects, and the decision may not be
    ``None``. The :attr:`decision` property of a SituationDecision is the decision; it may
    not be changed after the SituationDecision is created. The :attr:`attributes` property
    of a SituationDecision is a tuple of the attribute names of its situation. The values
    of those attributes may be retrieved as a tuple with the :attr:`situation` property.
    Individual attribute values may be retrieved with the :meth:`get` method, and changed
    with the :meth:`set` method. The agent that created a SituationDecision can be found
    using its :attr:`agent` property
    """

    def __repr__(self):
        return "<SituationDecision {} {:.20}>".format(
            self._decision, str(self._situation))

    def __init__(self, agent, decision, situation):
        if not isinstance(agent, Agent):
            raise IllegalArgumentError("{} is not an Agent".format(agent))
        if not isHashable(decision):
            raise IllegalArgumentError(
                "{} cannot be used as a decision".format(decision))
        for v in situation:
            ensureAttributeValue(v)
        self._agent = agent
        self._decision = decision
        ns = len(situation)
        na = len(self.agent._attributes)
        if ns == na:
            self._situation = list(situation)
        elif ns < na:
            self._situation = list(chain(situation,
                                                   repeat(None, na - ns)))
        else:
            raise IllegalArgumentError(
                "too many attribute values supplied for {} ({})".format(
                    self._agent, ns))

    @property
    def agent(self):
        """The ``Agent`` that created this SitutationDecision, and with which it remains associated."""
        return self._agent

    @property
    def decision(self):
        """The decision this SituationDecision represents.
        The decision is set when the SitutationDecision is created by an agent and
        may not be changed. It is typically a string or number, but may be any hashable object.
        """
        return self._decision

    @property
    def attributes(self):
        """A tuple of strings, the names of the attributes in the situation of this SituationDecision."""
        return self._agent.attributes

    @property
    def situation(self):
        """A tuple summarizing the situation this SituationDecision currently represents.
        Each attribute of the situation is an element of the returned tuple, in the same
        order as the :class"`Agent`'s attributes are represented in its :attr:`attributes`
        property. The value of an attribute must be hashable. The values of all the
        situtation's attributes can be set by setting this property to a tuple or list of
        the correct length. An :exc:`IllegalArgumentError` is raised if the tuple or list is
        longer than the number of attributes, or if any of the values supplied is not
        hashable. If the tuple or list supplied is shorter than the number of attributes
        in this SituationDecision the trailing ones are left unchanged. The values of
        individual attributes can be retrieved with the :meth:`get` method and changed with
        the :meth:`set` method of this SituationDecision.
        """
        return tuple(self._situation)

    # TODO Should we be leveraging Python's namedtuples instead of rolling
    #      our own here? It's not clear to me.
    @situation.setter
    def situation(self, values):
        if len(values) > len(self._agent._attributes):
            raise IllegalArgumentError(
                "{} has only {} attributes, but {} values were supplied".format(
                    self._agent, len(self._agent), len(values)))
        # we make a first pass all the way through the values so we
        # don't leave the situation only partially updated
        for v in values:
            ensureAttributeValue(v)
        for v, i in zip(values, count()):
            self._situation[i] = v

    def index(self, attribute):
        if isinstance(attribute, int):
            if attribute >= len(self._agent._attributes):
                raise IllegalArgumentError(
                    "{} has only {} attributes ({})".format(
                        self._agent, len(self._agent._attributes), attribute))
            elif attribute < 0:
                pass
            else:
                return attribute
        else:
            try:
                return self._agent._attributeIndices[attribute]
            except KeyError:
                pass
        raise IllegalArgumentError("{} is not an attribute of {}".format(
            attribute, self._agent))

    def get(self, attribute):
        """Returns value of the attribute named *attribute* of the situation of this SituatonDecision"""
        return self._situation[self.index(attribute)]

    def set(self, attribute, value):
        """Sets the value of the the attribute named *attribute* in this SituationDecisiion.
        The provided *value* must be hashable; if it is not an :exc:`IllegalArgumentError`
        is raised.
        """
        ensureAttributeValue(value)
        self._situation[self.index(attribute)] = value


class Instance:

    __slots__ = [ '_situation', '_decision', '_utility', '_reason', '_occurrences',
                  '_activationBase', '_activationNoise', '_activation',
                  '_retrievalProbability' ]

    def __repr__(self):
        return "<Instance {!r:.10} {:.25} {} ({:.15})>".format(
            self._decision, self._situation, self._utility, self._occurrences)

    def __init__(self, situation, decisionMade, outcomeProduced, reason):
        self._situation = situation
        self._decision = decisionMade
        self._utility = outcomeProduced
        self._reason = reason
        self._occurrences = []
        self._activationBase = None
        self._activationNoise = None
        self._activation = None
        self._retrievalProbability = None


class Closeable:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class Agent (Closeable):
    """A cognitive entity learning and making decisions based on its experience from prior decisions.
    The main entry point to PyIBL.
    An Agent has a *name*, a string, which can be retrieved with the :attr:`name`
    property. The name cannot be changed after an agent is created.
    If, when creating an agent, the *name* argument is not supplied or is ``None``,
    a name will be created of the form ``'Anonymous-Agent-n'``, where *n* is a
    unique integer.
    An agent also has zero or more *attributes*, named by strings. The attribute names
    can be retrieved with the :attr:`attributes` property, and also cannot be changed
    after an agent is created. Attempting to use a non-string (other than ``None`` for
    the *name*) for an agent or attribute name raises an :exc:`IllegalArgumentError`.

    The most important methods on agents are :meth:`choose` and :meth:`respond`.
    The choose method is typically called with two or more arguments,
    each a :class:`SituationDecision`.
    The agent chooses, based on its prior experiences, which of those
    SitutationDecisions will result in the greatest reward, and returns the
    corresponding :attr:`decision` from it.
    The respond method
    informs an agent of the result of its most recent choice;
    between any two requests that an agent make a choice the outcome
    of the earlier choice must be delivered to the agent by calling
    respond, or an :exc:`IllegalStateError` is raised.
    An outcome is a real (that is, not complex) number,
    where larger is implicitly "better" in some way.
    Before a PyIBL model learns from outcomes delivered to it by
    respond, the learning in the model must be bootstrapped
    in some way. There are two mechanisms for doing so:
    the :meth:`prepopulate` method and the :attr:`defaultUtility` property.
    Parameters controlling the learning model can be adjusted by setting
    the values of the :attr:`noise`, :attr:`decay` and :attr:`temperature`
    properties.
    If it is desired to use an agent multiple
    times it may be :meth:`reset`, erasing all memory of past
    interactions, though preserving its :attr:`noise`,
    :attr:`decay` and :attr:`temperature`  parameters, as well as its
    logging settings.

    Details of an agent's actions, and the computations underlying them,
    may be captured in a log. This is controlled by setting properties
    and calling methods on the agent's :class:`Population`, which can
    be retrieved at the value of the agent's :attr:`population` property.
    Alternatively most such logging methods and properties are also
    provided by Agent, which simply delegate to the agent's
    population. Like populations, agents can be used with Pythons
    ``with`` method to ensure the agent's population's log is close when
    exiting a block of code.
    """

    def __repr__(self):
        return "<Agent {} ({}, {}, {})>".format(
            self._name, len(self._attributes), len(self._instances), self._iteration)

    _nextId = 1

    def __init__(self, name=None, *attributes):
        if name is None:
            name = "Anonymous-Agent-{}".format(Agent._nextId)
            Agent._nextId += 1
        elif not (isinstance(name, str) and len(name) > 0):
            raise IllegalArgumentError("Agent name {} is not a non-empty string".format(name))
        self._name = name
        for a in attributes:
            if not (isinstance(a, str) and len(a) > 0):
                raise IllegalArgumentError("attribute name {} is not a non-empty string".format(a))
        d = {}
        for a, i in zip(attributes, count()):
            if a in d:
                raise IllegalArgumentError("duplicate attribute {}".format(a))
            d[a] = i
        self._attributes = attributes
        self._attributeIndices = d
        self.setPopulation(Population())
        self.noise = DEFAULT_NOISE
        self.temperature = None
        self.decay = DEFAULT_DECAY
        self.defaultUtility = None
        self.defaultUtilityPopulates = True
        self.reset()

    def reset(self):
        """Erases this agent's memory.
        It deletes all the instances from this agent, and arranges that the iteration
        and trial reported in logs are reset to 1. Logging details and destination, and
        IBLT parameters such as :attr:`noise` and :aatr:`decay` are not affected.
        Any prepopulated instances, including those created automatically if a
        :attr:`defaultUtility` is provided and :attr:`defaultUtilityPopulates` is true are
        removed, but the settings of those properties are not altered.
        """
        self._instances = OrderedDict()
        self._situationDecisions = []
        self._iteration = 0
        self._trial = 0
        self._pendingDecision = None
        self._logData = []

    def situationDecision(self, decision, *attributes):
        """Creates and returns a :class:`SitutationDecision` suitable for use with this agent.
        The decision is set to the value of the *decision8 argument, and may be any
        hashable value, though is typically a string or number. The *attributes* are the
        initial values of the various attributes of the situation, all of which must be
        hashable and are assigned in the order the attribute names were provided when
        creating the agent, which is also the same as the attribute names returned by
        the :attr:`attributes` property of this agent. An :exc:`IllegalArgumentError` is raised
        if more attribute values are provided than situtations for this agent support,
        or if the *decision* or any *attributes* are not hashable. If fewer values are
        provided they are used to initialize the first attributes of the situation,
        further attributes being initialized to ``None``.
        """
        n = len(self._attributes) - len(attributes)
        if n > 0:
            attributes = tuple(chain(attributes, repeat(None, n)))
        elif n < 0:
            raise IllegalArgumentError(
                "{} attribute values were supplied, but {} only has {} attributes".format(
                    len(attributes), self, len(self._attributes)))
        return SituationDecision(self, decision, attributes)

    @property
    def name(self):
        """The name of this Agent.
        It is a string, provided when the agent was created.
        """
        return self._name

    @property
    def attributes(self):
        """A tuple of the names of the attributes included in all situations associated with decisions this agent will be asked to make.
        These names are assigned when the agent is created and cannot be changed, and
        are strings. The order of them in the
        returned tuple is the same as that in which they were given when the agent was
        created, in which values are assigned when creating SitutationDecisions, and
        reported by the :attr:`situation` property of any SitutationDecisions created
        by this agent.
        """
        return self._attributes

    @property
    def population(self):
        """The :class:`Population` to which this Agent currently belongs.
        Every agent belongs to some Population. When an agent is freshly
        created it belongs to a freshly created Population that will be discarded if
        this agent is subsequently reassigned to a different population, by
        assigning to this property.
        Setting this property to ``None`` causes a new Population to be created
        and set as this agent's population.
        Attempting to assign to this property anything than a Population or ``None``
        raises an :exc:`IllegalArgumentError`.
        """
        return self._population

    @population.setter
    def population(self, value):
        if value is self._population:
            pass
        elif value is None:
            if len(self._population._agents) > 1:
                self.setPopulation(Population())
        elif isinstance(value, Population):
            if self._name in value._agentsByName:
                raise IllegalArgumentError("can't add Agent {} to Population {} because an Agent of that name is already present".format(self, value))
            self.setPopulation(value)
        else:
            # Be explicit, as Duck typing here could lead to inscrutable errors for the naive user
            raise IllegalArgumentError("{} is not a Population".format(value))

    def setPopulation(self, newPop):
        # For internal use only, assumes everything is of the correct types
        # Note that this is also called on an incompletely initialized Agent, that
        # does not yet have a _population.
        if hasattr(self, '_population') and self._name in self._population._agentsByName:
            self._population._agents.remove(self)
            del self._population._agentsByName[self._name]
        newPop._agentsByName[self._name] = self
        newPop._agents.append(self)
        self._population = newPop

    @property
    def noise(self):
        """The amount of noise to add during instance activation computation.
        This is typically a positive, possibly floating point, number between about 0.5 and 10.
        If zero, the default, no noise is added during activation computation.
        If set to ``None`` it reverts the value to its default, zero.
        If an explicit :attr:`temperature` is not set, the value of noise is also used
        to compute a default temperature for the value blending operation.
        """
        return self._noise

    @noise.setter
    def noise(self, value):
        if value is None:
            value = DEFAULT_NOISE
        if value != getattr(self, "_noise", None):
            self._noise = float(value)
            self._temperature = None

    @property
    def temperature(self):
        """The temperature parameter in the Boltzman Equation used for blending values.
        If ``none``, the default, the square root of 2 times the value of
        :attr:`noise` will be used. If ``None`` and there is also no noise, a waring
        will be printed and a default of 1 will be used for the temperature.
        """
        return self._temperatureParam

    @temperature.setter
    def temperature(self, value):
        self._temperatureParam = float(value) if value is not None else None
        self._temperature = None

    @property
    def decay(self):
        """Controls the rate at which activation for previously experienced instances in memory decay with the passage of time.
        Time in this sense is dimensionless, and simply the number choose/respond cycles that have occurred since the
        agent was created or last :meth:`reset`, as reported by the
        :ref:`tTrial or tIteration fields in log files <logging-strings>`. Typically a positive, possibly floating point,
        number between about 0.5 to about 10. If zero, the default, memory does not decay.
        If set to ``None`` it revert it reverts the value to its default, zero.
        """
        return self._decay

    @decay.setter
    def decay(self, value):
        if value is None:
            value = DEFAULT_DECAY
        if value != getattr(self, "_decay", None):
            self._decay = float(value)
            self._negdecay = -self._decay
            self._activationTermCache = list(repeat(None, max(len(getattr(self, "_activationTermCache", ())),
                                                                        ACTIVATION_CACHE_INCREMENT)))

    def close(self):
        """Calls the :meth:`Population.close` method of this agent's :attr:`population`.
        Note that calling this closes the log used by any other agents that belong to
        the same population.
        """
        self._population.close()

    @property
    def logging(self):
        """The :attr:`Population.logging` property of this agent's :attr:`population`.
        May be assigned to, in which case it sets the agent's population's logging
        property. Note that so assigning to this affects the logging behavior of any other
        agents that belong to the same population.
        """
        return self._population.logging

    @logging.setter
    def logging(self, value):
        self._population.logging = value

    def logToList(self, lst=None):
        """Calls the :meth:`Population.logToList` method of this agent's :attr:`population`, with the same *lst* argument.
        Note that calling this affects the logging behavior of any other agents that
        belong to the same population.
        """
        return self._population.logToList(lst)

    def logToFile(self, file,
                  heading=LOG_HEADING_DEFAULT,
                  precision=DEFAULT_PRECISION,
                  dialect=DEFAULT_CSV_DIALECT):
        """Calls the :meth:`Population.logToFile` method of this agent's :attr:`population`, with the same arguments.
        Note that calling this affects the logging behavior of any other agents that
        belong to the same population.
        """
        return self._population.logToFile(file, heading, precision, dialect)

    def logToDatabase(self, database, table,
                      create=CREATE_DB_DEFAULT,
                      clear=CLEAR_DB_DEFAULT):
        """Calls the :meth:`Population.logToDatabase` method of this agent's :attr:`population`, with the same arguments.
        Note that calling this affects the logging behavior of any other agents that
        belong to the same population.
        """
        return self._population.logToDatabase(database, table, create, clear)

    @property
    def block(self):
        """The :attr:`Population.block` property of this agent's :attr:`population`.
        May be assigned to, in which case it sets the agent's population's block
        property. Note that so assigning to this affects the logging behavior of any other
        agents that belong to the same population.
        """
        return self._population.block

    @block.setter
    def block(self, value):
        self._population.block = value

    @property
    def occurrencesLimit(self):
        """The :attr:`Population.occurencesLimit` property of this agent's :attr:`population`.
        May be assigned to, in which case it sets the agent's population's occurrencesLimit
        property. Note that so assigning to this affects the logging behavior of any other
        agents that belong to the same population.
        """
        return self._population.occurrencesLimit

    @occurrencesLimit.setter
    def occurrencesLimit(self, value):
        self._population.occurrencesLimit = value

    @property
    def defaultUtility(self):
        """The utility, or a function to compute the utility, if there is no matching instance for a :class:`SituationDecision`.
        When :meth:`choose` is called, for each SituationDecision passed to it it is
        first ascertained whether or not an instance exists that matches that decision
        in the given situation. If there is not, the value of this property is consulted.
        Note that an instance added with :meth:`prepopulate` counts as matching, and will
        prevent the interrogation of this property.

        The value of this property may be a number, in which case when needed it is
        simply used as the default utility. If it is not a number, it is assumed to
        be a function that takes one argument, a SituationDecion. When a default utility is
        needed that function will be called, passing the SituationDecision in question
        to it, and value returned, which should be a number, will be used.
        If at that time the value is not a function of one argument, or it does not
        return a number, an :exc:`IllegalStateError` is raised.

        The :attr:`defaultUtilityPopulates` property, which is ``True`` by default,
        controls whether or not an instance is added for each interrogation of
        the attr:`defaultUtility` property. If an instance is added, it is added
        as by :meth:`prepopulate`. Note that, except for the first interrogation of
        this property, such added instances will have timestamps greater than zero.

        Setting this property to ``None`` or ``False`` causes no default probability
        to be used. In this case, if :meth:`choose` is called for a decision in a
        situation for which there is no instance available, an :exc:`IllegalStateError`
        will be raised.
        """
        return self._defaultUtility

    @defaultUtility.setter
    def defaultUtility(self, value):
        if value is None:
            self._callableDefaultUtility = False
        else:
            try:
                value <= 1      # value doesn't matter, just ensuring comparable
                self._callableDefaultUtility = False
            except TypeError:
                self._callableDefaultUtility = True
        self._defaultUtility = value

    @property
    def defaultUtilityPopulates(self):
        """Whether or not a default utility provided by the :attr:`defaultUtility` property is also entered as an instance in memory.
        This property has no effect is defaultUtility is ``None`` or ``False``.
        """
        return self._defaultUtilityPopulates

    @defaultUtilityPopulates.setter
    def defaultUtilityPopulates(self, value):
        self._defaultUtilityPopulates = bool(value)

    def prepopulate(self, outcome, *situationDecisions):
        """Adds instances to memory, one for each of the *situationDecisions*, with the given outcome, at the current time, without advancing that time.
        Time is a dimensionless quantity, simply a count of the number of choose/respond
        cycles that have occurred since the agent was created or last :meth:`reset`, and
        is report in logs using the :ref:`tTrial or tIteration columuns
        <logging-strings>`.

        This is typically used to enable startup of a model by adding instances before the
        first call to :meth:`choose`. When used in this way the timestamp associated with
        this occurrence of the instance will be zero. Subsequent occurrences are possible
        if :meth:`respond` is called with the same outcome after :meth:`choose` has
        returned the same decision in the same situation, in which case those reinforcing
        occurrences will have later timestamps. An alternative mechanism to facility
        sartup of a model is setting the :attr:`defaultUtility` property of the agent.
        While rarely done, a modeler can even combine the two mechanisms, if desired.

        It is also possible to call prepopulate after choose/respond cycles have occurred.
        In this case the instances are added with the current time as the timestamp. This
        is one less than the timestamp that would be used were an instance to be added by
        being experienced as part of a choose/respond cycle instead. Each agent keeps
        internally a clock, the number of choose/respond cycles that have occurred since
        it was created or last :meth:`reset`. When :meth:`choose` is called it advances
        that clock by one *before* computing the activations of the existing instances, as
        it must as the activation computation depends upon all experiences having been in
        the past. That advanced clock is the timestamp used when an instance is added or
        reinforced by :meth:`respond`. If an attempt is made to add a prepopulated
        instance for a decision in a situation at a time for which an instance has
        already occurred a warning will be issued, but the instance will still be added
        or reinformaced as appropriate.
        """
        ensurePossibleOutcome(outcome)
        for sd in situationDecisions:
            sd = self.canonicalizeSituationDecision(sd)
            if self.addInstance('prepopulated', outcome, sd.decision, sd.situation):
                pyiblwarn(
                    "there is already an instance for {} in {}".format(
                        sd, self))

    def canonicalizeSituationDecision(self, sd):
        if isinstance(sd, SituationDecision):
            return sd
        try:
            ensurePossibleDecision(sd)
            return self.situationDecision(sd)
        except:
            pass
        raise IllegalArgumentError("{} is not a SituationDecision".format(sd))

    SQRT2 = math.sqrt(2)
    HASH_LIMIT = -2**hash_info.width # less than any possible hash value

    _arbitrarySelection = 0

    def choose(self, *situationDecisions):
        """Selects which of the *selectionDecisions* is expected to result in the largest payoff, and returns its decision.
        Each of the *situationDecisions* must have a different
        :attr:`SituationDecision.decision`.
        The situations of the *situuationDecisions* must have the same attributes
        as this agent, and will typically have been created
        by its :meth:`siutationDecision` method.
        If any deciions are duplicated, or if any of the *situationDecisions*'s attributes
        do not match those of this agent, an :exc:`IllegalArgumentError` is raised.

        It is also possible to supply no *situationDecisions*, in which case those used
        in the most recent previous call to this method are reused. If there was no
        previous call to choose, an :exc:`IllegalArgumentError` is raised.

        As a convenience decisions may be passed as arguments insteaad of one or more
        of the *situationDecisions*. As in :class:`SituationDecision` such a decision
        must be a Python object that is hashable, but is not ``None``. For each such
        decision passed a SituationDecision will be implictly constructed with the
        decision and with all its attributes having the value ``None``. If any decisions
        cited are not hashable, are ``None``, or duplicate other decisions cited,
        including those cited in a :attr:`SituationDecision`, an
        :exc:`IllegalArgumentError` is raised.

        For each of the *situationDecisions* it finds all instances in memory for the
        same decision in the same situation, and computes their activations at the current
        time based upon when in the past they have been seen, modified by the value of the
        :attr:`decay` property, and with noise added as controlled by the
        :attr:`noise` property. Looking at the activations of the whole ensemble of
        matching instances a retrieval probability is computed for each, and these are
        combined to arrive at blended value expected for each decision. This blending
        operation depends upon the value :attr:`temperature` property; if none is
        supplied a default is computed based on the :attr:`noise`.
        The decision chosen and returned is that with the highest blended value.
        In case of a tie, if :attr:`noise` is non-zero one will be chosen at random, and
        otherwise an arbitrary, but deterministic, choice is made among tied decisions.
        Note that the return value is the decision from the SitutaionDecision, and
        not the whole SituationDecision.

        After a call to choose a corresponding response must be delivered to the agent
        with :meth:`respond` before calling choose again, or an :exc:`IllegalStateError`
        will be raised.
        """
        if self._pendingDecision:
            # in the future we will probably allow delayed feedback, but not yet
            raise IllegalStateError("decision requested before previous outcome supplied")
        if situationDecisions:
            self._situationDecisions = situationDecisions
        elif self._situationDecisions:
            situationDecisions = self._situationDecisions
        else:
            raise IllegalArgumentError("no SituationDecisions supplied")
        situationDecisions = tuple(map(self.canonicalizeSituationDecision, situationDecisions))
        # TODO is there a more Pythonic way to initialize sdMap?
        sdMap = OrderedDict()
        sSet = set()
        for sd in situationDecisions:
            sdMap[sd.decision] = sd.situation
            sSet.add(sd.situation)
        if len(sdMap) < len(situationDecisions):
            raise IllegalArgumentError(
                "Duplicate possible decisions supplied {}".format(
                    tuple(d for d, n in Counter(sdMap.keys()).items() if n > 1)))
        if self._temperature is None:
            if not self._temperatureParam:
                if abs(self._noise) < ZERO_NOISE_TOLERANCE:
                    pyiblwarn(("There is no noise nor explicit temperature, so the temperature " +
                               "is being set to {}.").format(ZERO_NOISE_DEFAULT_TEMPERATURE))
                    self._temperature = ZERO_NOISE_DEFAULT_TEMPERATURE
                else:
                    self._temperature = self._noise * Agent.SQRT2
            else:
                self._temperature = self._temperatureParam
            if abs(self._temperature) < MINIMUM_TEMPERATURE:
                pyiblwarn(("The absolute value of the temperature or noise ({}, {}) is so small that " +
                           "it falls outside the range that IBL theory supports, and may result in " +
                           "division by zero errors.").format(self._temperatureParam, self._noise))
        # The following two loops will have to change radically when we do partial
        # matching. But for now....
        for sd in situationDecisions:
            if not (sd.decision in self._instances and sd.situation in self._instances[sd.decision]):
                if self._callableDefaultUtility:
                    try:
                        utility = self._defaultUtility(sd)
                        if not isinstance(utility, Number):
                            raise IllegalStateError(
                                "default utility function returned a non number: {}".format(
                                    utility))
                    except:
                        raise IllegalStateError(
                            "default utility function failed to return a value")
                elif self._defaultUtility is not None:
                    utility = self._defaultUtility
                    try:
                        utility > 0
                    except:
                        raise IllegalStateError(
                            "default utility is a non number: {}".format(utility))
                else:
                    raise IllegalStateError(
                        "no utility for {}, either set a default utility or add a suitable prepopulated instance".format(
                            sd))
                self.addInstance('defaulted', utility,  sd.decision, sd.situation)
        self._iteration += 1
        self._trial += 1
        maxBlendedValue = float("-inf")
        bestChoices = []
        blendedValues = {} if 'oBlendedValue' in self._population._logging else None
        logUnusedDecisions = (self._population._logLevel >= LOG_INSTANCES and
                              'unusedDecisions' in self._population._logging)
        logUnusedSituations = (self._population._logLevel >= LOG_INSTANCES and
                              'unusedSituations' in self._population._logging)
        for decision, bySituation in self._instances.items():
            if decision in sdMap:
                optionSituation = sdMap[decision]
                for situation, byOutcome in bySituation.items():
                    if situation == optionSituation:
                        blendedValue = self.blendedValue(byOutcome.values())
                        assert blendedValue is not None
                        if blendedValues is not None:
                            blendedValues[decision] = blendedValue
                        if blendedValue < maxBlendedValue:
                            continue
                        elif blendedValue > maxBlendedValue:
                            maxBlendedValue = blendedValue
                            bestChoices.clear()
                        bestChoices.append(decision)
                    elif logUnusedSituations:
                        for inst in byOutcome.values():
                            self.activation(inst)
                            inst._retrievalProbability = None
            elif logUnusedDecisions:
                for situation, byOutcome in bySituation.items():
                    if logUnusedSituations or situation in sSet:
                        for inst in byOutcome.values():
                            self.activation(inst)
                            inst._retrievalProbability = None
        assert bestChoices
        if len(bestChoices) == 1:
            result = bestChoices[0]
        else:
            if self._noise:
                result = random.choice(bestChoices)
            else:
                # make an arbitrary but deterministic selection
                result = bestChoices[Agent._arbitrarySelection % len(bestChoices)]
                Agent._arbitrarySelection += 1
        self._pendingDecision = (result, sdMap[result])
        if self._population._logging:
            self.logEntries(sdMap, sSet, result, blendedValues)
        return result

    def logItem(self, detail, value, lst):
        lst[self._population._logPositions[detail._index]] = value

    def logTest(self, detail):
        return detail in self._population._logDetails

    def testAndLogItem(self, detail, value, lst):
        if self.logTest(detail):
            self.logItem(detail, value, lst)

    def logAppend(self, data, sequenceIndex):
        self._population._logSequence += 1
        if sequenceIndex is not None:
            data[sequenceIndex] = self._population._logSequence
        self._logData.append(data)

    def logEntries(self, sdMap, sSet, result, blendedValues):
        pop = self._population
        prototype = [None] * pop._logWidth
        pos = pop._logPositions
        details = pop._logging
        sequenceIndex = logIndecies['sequence'] if 'sequence' in details else None
        if 'tAgent' in details:
            prototype[pos[logIndecies['tAgent']]] = self._name
        if 'tBlock' in details:
            prototype[pos[logIndecies['tBlock']]] = pop._block
        if 'tTrial' in details:
            prototype[pos[logIndecies['tTrial']]] = self._trial
        if 'tIteration' in details:
            prototype[pos[logIndecies['tIteration']]] = self._iteration
        if 'tNoiseParam' in details:
            prototype[pos[logIndecies['tNoiseParam']]] = self._noise
        if 'tTemperature' in details:
            prototype[pos[logIndecies['tTemperature']]] = self._temperature
        if 'tDecayParam' in details:
            prototype[pos[logIndecies['tDecayParam']]] = self._decay
        if pop._logLevel >= LOG_OPTIONS:
            logUnusedDecisions = (self._population._logLevel >= LOG_INSTANCES and
                                  'unusedDecisions' in self._population._logging)
            logUnusedSituations = (self._population._logLevel >= LOG_INSTANCES and
                                  'unusedSituations' in self._population._logging)
            for decision, bySituation in self._instances.items():
                byOption = list(prototype)
                if 'oDecision' in details:
                    byOption[pos[logIndecies['oDecision']]] = decision
                for situation, byOutcome in bySituation.items():
                    isCurrentOption = decision in sdMap and sdMap[decision] == situation
                    if 'oSituation' in details:
                        byOption[pos[logIndecies['oSituation']]] = situation
                    if 'oBlendedValue' in details:
                        byOption[pos[logIndecies['oBlendedValue']]] = blendedValues[decision] if isCurrentOption else None
                    if pop._logLevel >= LOG_INSTANCES and (isCurrentOption or
                                                           (logUnusedDecisions and logUnusedSituations) or
                                                           (logUnusedSituations and decision in sdMap) or
                                                           (logUnusedDecisions and situation in sSet)):
                        for instance in byOutcome.values():
                            byInstance = list(byOption)
                            if 'iUtility' in details:
                                byInstance[pos[logIndecies['iUtility']]] = instance._utility
                            if 'iReasonAdded' in details:
                                byInstance[pos[logIndecies['iReasonAdded']]] = instance._reason
                            if 'iFrequency' in details:
                                byInstance[pos[logIndecies['iFrequency']]] = len(instance._occurrences)
                            if 'iOccurrences' in details:
                                if pop._occurrencesLimit:
                                    byInstance[pos[logIndecies['iOccurrences']]] = ",".join(map(str, instance._occurrences[-pop._occurrencesLimit:]))
                                else:
                                    byInstance[pos[logIndecies['iOccurrences']]] = ",".join(map(str, instance._occurrences))
                            if 'iActivationBase' in details:
                                byInstance[pos[logIndecies['iActivationBase']]] = instance._activationBase
                            if 'iActivationNoise' in details:
                                byInstance[pos[logIndecies['iActivationNoise']]] = instance._activationNoise
                            if 'iActivation' in details:
                                byInstance[pos[logIndecies['iActivation']]] = instance._activation
                            if 'iRetrievalProbability' in details:
                                byInstance[pos[logIndecies['iRetrievalProbability']]] = instance._retrievalProbability
                            self.logAppend(byInstance, sequenceIndex)
                    elif isCurrentOption:
                        self.logAppend(byOption, sequenceIndex)
        else:
            self.logAppend(prototype, sequenceIndex)

    def respond(self, outcome, flush=True, close=False):
        """Provide the *outcome* resulting from the most recent decision returned by :meth:`choose`.
        The *outcome* should be a non-complex number, where larger numbers are considered "better."
        This results in the creation or reinforcemnt of an instance in memory for the
        decision, in the situation it had when chosen, with the given outcome, and is
        the fundamental way in which the PyIBL model "learns from experience."

        If there has not been a call to choose since the last time respond was called an
        :exc:`IllegalStateError` is raised. If *outcome* is not a non-complex number an
        :exc:`IllegalArgumentError` is raised.

        If *flush* is not false and this agent's :attr:`population` has logging enabled,
        any logging information still buffered at the conclusion of respond's actions will
        be written to disk, the database, or the open stream, as appropriate; *flush* has
        no special effect if the log is being written to a list. If *close* is not false
        the log file or database will be closed; it has no effect on an open stream passed
        to :meth:`Population.logToFile` nor on logging to a list. Note that if a log file
        or database is so closed, it will automatically be reopened next time there is a
        need to write to it.
        """
        ensurePossibleOutcome(outcome)
        if not self._pendingDecision:
            raise IllegalStateError("outcome {} supplied when no decision requiring an outcome has been supplied".format(outcome))
        self.addInstance('experienced', outcome, *self._pendingDecision)
        pop = self._population
        if pop._logging:
            pos = pop._logPositions
            details = pop._logging
            for row in self._logData:
                if 'tChoice' in details:
                    row[pos[logIndecies['tChoice']]] = self._pendingDecision[0]
                if 'tChoiceSituation' in details:
                    row[pos[logIndecies['tChoiceSituation']]] = self._pendingDecision[1]
                if 'tResponse' in details:
                    row[pos[logIndecies['tResponse']]] = outcome
            pop._logger.log(self._logData)
            self._logData.clear()
            if flush:
                pop._logger.flush()
            if close:
                # Note that this is different than closing the population, as
                # it just closes the logger, but does not reset the population's
                # agents.
                pop._logger.close()
        self._pendingDecision = None

    # Typically this method should not be overriden. Override computeActivationBase and
    # computeActivationNoise instead.
    def activation(self, instance):
        result = self.computeActivationBase(instance)
        instance._activationBase = result
        if self._noise:
            noise = self.computeActivationNoise()
            instance._activationNoise = noise
            result += noise
        instance._activation = result
        return result

    # homonymy warning: math.log, used several times below, is a logarithm, and unrelated to logging

    # This is deliberately not a method of Instance, so it can easily be overriden by subclassing Agent.
    def computeActivationBase(self, instance):
        # TODO consider short-circuiting the following when the contribution gets too small
        # ugly, stupid hand optimation follows, but this is the inner loop
        cache = self._activationTermCache
        cacheLen = len(cache)
        result = 0.0
        now = self._iteration
        for n in instance._occurrences:
            if n >= now:
                break
            t = now - n
            if t >= cacheLen:
                newLen = t + ACTIVATION_CACHE_INCREMENT
                cache.extend(repeat(None, newLen - cacheLen))
                cacheLen = newLen
            term = cache[t]
            if term is None:
                cache[t] = term = t ** self._negdecay
            result += term
        return math.log(result) if result else DEFAULT_ACTIVATION

    # This is deliberately not a method of Instace, so it can easily be overriden by subclassing Agent.
    def computeActivationNoise(self):
        p = random.uniform(0.0001, 0.9999)
        return self._noise * math.log((1 - p) / p)

    # Typically this method should not be overriden. Override computeRetrievalProbabilities instead.
    def retrievalProbabilities(self, instances):
        result = self.computeRetrievalProbabilities(instances,
                                                    (self.activation(i) for i in instances))
        for i, p in zip(instances, result):
            i._retrievalProbability = p
        return result

    def computeRetrievalProbabilities(self, instances, activations):
        # instances is ignored, but is passed in case an overriding implementation needs it
        probabilities = [math.exp(activation / self._temperature) for activation in activations]
        total = sum(probabilities)
        return [p / total for p in probabilities] if total else probabilities

    # Typically this method should not be overriden. Override computeBlendedValue instead.
    def blendedValue(self, instances):
        return self.computeBlendedValue(instances, self.retrievalProbabilities(instances))

    def computeBlendedValue(self, instances, probabilities):
        return sum([instance._utility * probability for instance, probability in zip(instances, probabilities)])

    def addInstance(self, reason, outcome, decision, situation):
        # TODO The structure this imposes on self._instances is almost
        #      surely *not* the right one long term, for when we do
        #      similarity and blending. But for now....
        result = True
        if decision in self._instances:
            bySituation = self._instances[decision]
        else:
            bySituation = OrderedDict()
            self._instances[decision] = bySituation
        if situation in bySituation:
            byOutcome = bySituation[situation]
        else:
            byOutcome = OrderedDict()
            bySituation[situation] = byOutcome
            result = False
        # TODO Should instances with outcomes differing only by less than some epsilon be coalesced?
        if outcome in byOutcome:
            instance = byOutcome[outcome]
        else:
            instance = Instance(situation, decision, outcome, reason)
            byOutcome[outcome] = instance
        instance._occurrences.append(self._iteration)
        return result

    def showLog(self, destination=stdout):
        self._population.showLog(destination)

    def showInstances(self, file=stdout, limit=1000, precision=4):
        tab = PrettyTable([
            'Decision', 'Situation', 'Occurences',
            'Activation base', 'Activation noise', 'Activation',
            'Retrieval probability', 'Utility'
         ])
        tab.align = 'r'
        fmt = ".{}f".format(precision)
        try:
            for u in self._instances.values():
                for v in u.values():
                    for i in v.values():
                        limit -= 1
                        if limit <= 0:
                            raise StopIteration()
                        tab.add_row([
                            i._decision, i._situation, i._occurrences,
                            _safeFormat(i._activationBase, fmt),
                            _safeFormat(i._activationNoise, fmt),
                            _safeFormat(i._activation, fmt),
                            _safeFormat(i._retrievalProbability, fmt),
                            i._utility
                        ])
            raise StopIteration()
        except StopIteration:
            pyiblwarn(
                """The showInstances() method shows the current, internal state of the
                instances. The activations and retrieval probabilities therein may be very
                old or non-existant. This method should be used only as a debugging tool:
                for accurate information about the "mental" state of the model use PyIBL's
                logging facilities instead.""")
            print("{}: iteration {}, trial {}.".format(
                self._name, self._iteration, self._trial))
            print(tab, file=file, flush=True)


class Population (Closeable):

    """A collection of :class:`Agent` objects, sharing logging information.
    Every Agent belongs to exactly one Population, and all the agents within a
    population share the same log, if one is enabled. Methods for configuring logs apply
    to populations. Log related methods of agents actually apply to that agent's
    population and so affect the logging behavior of all other agents that belong to that
    population. There is a fixed order of the agents in a population, the order they
    were added to it by setting their :attr:`population` property. A number of methods
    on Population simply delegate to a population's agents, being called on each
    agent, in that fixed order.

    Logging is turned on, and the details of what information is to be included configured,
    by setting the :attr:`logging` property. Where to write the log is determinded by
    calling on of the methods :meth:`LogToFile`, :meth:`LogToDatabase` or
    :meth:`LogToList`. The :attr:`agents` property returns a tuple of the agents that
    are a part of a population. If a population opens a log file (as opposed to
    receiving an already open file) or database, it can be closed by calling the
    population's :meth:`close` method. A population can be used with Pythons ``with``
    method to ensure that close is called when leaving a block of code, and such
    use is recommended.
    """

    def __repr__(self):
        return "<Population {}>".format(", ".join(a.name for a in self._agents)[0:60])

    def __init__(self):
        self._agents = []
        self._agentsByName = {}
        self.logToFile(None)
        self.logging = None
        self.block = None
        self.occurrencesLimit = None

    def addAgents(self, *agents):
        # TODO delete this obsolete method, retained temporarily to support
        #      the exisiting cyber warfare example code
        for agent in agents:
            agent.population = self

    def ensureAgents(self):
        if not self._agents:
            raise IllegalStateError("the Population has no Agents")

    @property
    def agents(self):
        """Returns a tuple of all the Agents currently in this Population.
        The order is the fixed order that they will be traversed by operations operating
        on all of them."""
        return tuple(self._agents)

    def resetAgents(self):
        """Calls the :meth:`Agent.reset` method of all the Agents in this Population.
        Raises an :exc:`IllegalStateError` if this Population has no agents.
        """
        self.ensureAgents()
        for agent in self._agents:
            agent.reset()

    def setNoise(self, value):
        """Sets the :attr:`Agent.noise` property of all the Agents in this Population to *value*.
        Raises an :exc:`IllegalStateError` if this Population has no agents.
        """
        self.ensureAgents()
        for agent in self._agents:
            agent.noise = value

    def setTemperature(self, value):
        """Sets the :attr:`Agent.temperature` property of all the Agents in this Population to *value*.
        Raises an :exc:`IllegalStateError` if this Population has no agents.
        """
        self.ensureAgents()
        for agent in self._agents:
            agent.temperature = value

    def setDecay(self, value, *agentNames):
        """Sets the :attr:`Agent.decay` property of all the Agents in this Population to *value*.
        Raises an :exc:`IllegalStateError` if this Population has no agents.
        """
        self.ensureAgents()
        for agent in self._agents:
            agent.decay = value

    def setDefaultUtility(self, value):
        """Sets the :attr:`Agent.defaultUtility` property of all the Agents in this Population to *value*.
        Raises an :exc:`IllegalStateError` if this Population has no agents.
        """
        self.ensureAgents()
        for agent in self._agents:
            agent.defaultUtility = value

    def prepopulate(self, outcome, *situationDecisions):
        """Calls the :meth:`Agent.prepopulate` method of all the Agents in this Population, with the given *outcome* and *situationDecisions*.
        Raises an :exc:`IllegalStateError` if this Population has no agents.
        """
        self.ensureAgents()
        for agent in self._agents:
            agent.prepopulate(outcome, *situationDecisions)

    def setDefaultUtilityPopulates(self, value):
        """Sets the :attr:`Agent.defaultUtilityPopulates` property of all the Agents in this Population to *value*.
        Raises an :exc:`IllegalStateError` if this Population has no agents.
        """
        self.ensureAgents()
        for agent in self._agents:
            agent.defaultUtilityPopulates = value

    def close(self):
        """If a log file or database has been opened by this Population, it flushes and then closes it.
        If an open stream was passed to this Population for logging it is flushed, but not closed.
        It also calls the :meth:`Agent.reset` method of all the Agents in this Population.
        No error is raised if there is no log open, though any agents are still reset.
        """
        if self._logger:
            self._logger.flush()
            self._logger.close()
        for agent in self._agents:
            agent.reset()

    @property
    def logging(self):
        """A set of :ref:`strings describing columns to be added to the log <logging-strings>`.
        As a convenience assigning a string to this property is equivalent to assigning
        a single element set containing just that string to it.
        Any other iterable of strings may also be assigned to this property, it being
        equivalent to assigning a set of those strings to it.
        If an empty set, or ``None``, is assigned to this property it turns logging
        off. The default value of this property in a newly created population is ``None``.
        If any of the strings assigned to this property are not among
        :ref:`those that name logging columns <logging-strings>` an
        :exc:`IllegalArgumentError` is raised.
        """
        return self._logging

    @logging.setter
    def logging(self, value):
        if value is None:
            value = frozenset()
        elif isinstance(value, str):
            value = frozenset({value})
        else:
            value = frozenset(value)
        unknowns = tuple(v for v in value if v not in logDetails)
        if unknowns:
            raise IllegalArgumentError("unknown log details {}".format(unknowns))
        self._logging = value
        self._logLevel = max(logLevels[v] for v in value) if value else 0
        columns = tuple(d for d in logDetails if d in value and d in logIndecies)
        self._logWidth = len(columns)
        self._logPositions = [None]*logCount
        for c, i in zip(columns, count()):
            self._logPositions[logIndecies[c]] = i
        self.close()
        if self._logger:
            self._logger.reset()

    def setLogger(self, kind, *args):
        if getattr(self, "_logger", None):
            self._logger.close()
        self._logger = globals()[kind](self, *args)
        return args[0]

    def logToList(self, destination=None):
        """Arranges for this population's log to be "written" to a list in memory, for possible future manipulation.
        Returns the list to which logging information will be added. For each "row" of the
        log a sublist will be added to this list, and each "column" will be added as an
        element of that sublist. Unlike when writing to a file the objects added to the
        sublists are never converted to strings or otherwise reformatted. If *destination*
        is supplied it should be the list to which logging information is added; if it is
        ``None``, the default, a new, empty list will be created and used. The
        *destination* need not actually be a list, it can be any object that implements
        the **extend()** method. If the value of *destination* does not implement
        that method, and is not `None`, an :exc:`IllegalArgumentException` is raised.
        """
        if destination is None:
            destination = []
        if not hasattr(destination, "extend"):
            raise IllegalArgumentError("{} is not a list and does not otherwise implement extend()".format(destination))
        return self.setLogger("ListLogger", destination)

    def logToFile(self, file,
                  heading=LOG_HEADING_DEFAULT,
                  precision=DEFAULT_PRECISION,
                  dialect=DEFAULT_CSV_DIALECT):
        """Arranges for this population's log to be written to a file.
        If *file* is ``None``, or another false value, the standard output is used.
        If *file* is a file object, or any other object implmenting the **write()**
        method, it is used as is. Otherwise is is converted to a string and used as
        a filename, which is opened for writing; the contents of any existing file
        of that name are lost. The :meth:`close` method will only close a file that
        has been opened by this population. If an already open file, or ``None``, was
        pass as the *file* argument this population will never attempt to close it.

        The file is typically written in Commad Separated Values (CSV) format. The
        format can be change by providing a different *dialect*, a string. See the
        documentation of Python's
        `csv module <https://docs.python.org/3.4/library/csv.html>`_ for details of
        what dialects are available.
        If *heading* is not false when first writing to this file a header row is
        written.
        Floating point numbers are rounded to the given *precision*, which defaults
        to four digits after the decimal point.
        """
        heading = bool(heading)
        if precision <= MINIMUM_PRECISION:
            precision = MINIMUM_PRECISION
        elif precision >= MAXIMUM_PRECISION:
            precision = MAXIMUM_PRECISION
        else:
            precision = round(precision)
        if not file:
            return self.setLogger("StreamLogger", stdout, heading, dialect, precision)
        elif hasattr(file, "write"):
            return self.setLogger("StreamLogger", file, heading, dialect, precision)
        else:
            return self.setLogger("FileLogger", str(file), heading, dialect, precision)

    def logToDatabase(self, database, table,
                      create=CREATE_DB_DEFAULT,
                      clear=CLEAR_DB_DEFAULT):
        """Arranges for this population's log to be written to a table in a `SQLite <http://www.sqlite.org/>`_ database.
        The *database* can be a string, in which case a database of that name is opened;
        or it can be a connection object which is used as is. The *table* is converted
        to a string and used as the name of the table into which to write log rows.
        If the boolean *create* is true (it is false by default) a table of the given name
        will be created with appropriate columns; otherwise it should exist already.
        if the boolean *clear* is true (it is false by default) when starting a new log
        all existing rows in the table will be deleted; otherwise the new rows are added
        to those already present.
        """
        return self.setLogger("DatabaseLogger", database, str(table), bool(create), bool(clear))

    @property
    def block(self):
        """A hashable Python object that will be written to a log in the ``'tBlock'`` column.
        This is typically useful for keeping track of participants or experimental conditions.
        Multiple values can easily be multiplexed into this column by setting this
        property to a tuple. If block is being used as an integer counter a particularly
        easy way to manipulate it may be with the :const:`NEXT` constant. If an attempt is made
        to set this property to a non-hashable object, other than :const:`NEXT`, an
        :exc:`IllegalArgumentError` is raised.
        """
        return self._block

    @block.setter
    def block(self, value):
        # we insist the value be hashable so we don't have to worry about when it might
        # change out from under us versus when it gets written to the log
        for agent in self._agents:
            agent._trial = 0
        if value is NEXT:
            try:
                self._block += 1
            except TypeError:
                self._block = 1
        else:
            try:
                hash(value)         # raise an error if not hashable
                self._block = value
            except:
                raise IllegalArgumentError("block value {} is not hashable".format(value))

    @property
    def occurrencesLimit(self):
        """The maximum number of values to include in the iOccurrences field of a log, or ``None`` if there is no limit."""
        return self._occurrencesLimit

    @occurrencesLimit.setter
    def occurrencesLimit(self, value):
        try:
            self._occurrencesLimit = int(value) if value and value > 0 else None
            return
        except:
            raise IllegalArgumentError("{} is not a legal occurrencesLimit".format(value))

    def choose(self, *situationDecisions):
        """Calls the :meth:`Agent.choose` method of each agent in this population, and returns a tuple of the resulting decisions.
        The *situationDecisions* are all passed, in the same order, too all of the agents.
        The calls to choose are made on the agents in the order in which the agents
        appear in the population's :attr:`agents` property, which is the order they were
        added to the population. Note that for it to be possible to use this method
        all the agents in the population must expect exactly the same attributes.
        If this population has no agents an :exc:`IllegalStateError` is raised.
        """
        self.ensureAgents()
        return tuple(agent.choose(*situationDecisions) for agent in self._agents)

    def respond(self, outcomes, flush=True, close=False):
        """Calls the :meth:`Agent.respond` method of each agent in this population, passing them the values in *outcomes*.
        The agents are responded to in the order in which they were added to this
        population, the same order they appear in the :attr:`agents` property, and the
        values chosen from *outcomes* are selected in parallel in this same order.
        The *outcomes* should be an iterable of non-complex numbers.

        If *flush* is not false and this population has logging enabled, any logging
        information still buffered at the conclusion of the last respond's actions will be
        written to disk, the database, or the open stream, as appropriate; *flush* has no
        special effect if the log is being written to a list. If *close* is not false the
        log file or database will be closed; it has no effect on an open stream passed to
        :meth:`logToFile` nor on logging to a list. Note that if a log file or database is
        so closed, it will automatically be reopened next time there is a need to write to
        it.

        An :exc:`IllegalArgumentError` is raised if *outcomes* does not contain the same
        number of values as this population contains agents.
        """
        self.ensureAgents()
        outcomes = list(outcomes) # in case outcomes is an iterable but not a sequence
        if len(outcomes) != len(self._agents):
            raise IllegalArgumentError("{} outcomes supplied, but {} contains {} agents".format(
                len(outcomes, self, len(self._agents))))
        for (agent, outcome) in zip(self._agents, outcomes):
            agent.respond(outcome, False)
        if flush:
            self._logger.flush()
        if close:
            self._logger.close()

    def showLog(self, destination=stdout):
        # TODO figure out how to die more gracefully if the log is huge
        self._logger.close()
        table = self._logger.getPrettyTable()
        table.align = 'r'
        print(table, file=destination)


class Logger:
    # abstract base class

    def __init__(self, population):
        self._population = population
        self._started = False
        population._logSequence = 0

    def log(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        self.flush()

    def reset(self):
        self.close()
        self._started = False
        self._population._logSequence = 0

    def getPrettyTable(self):
        raise NotImplementedError()


class ListLogger(Logger):

    def __init__(self, population, lst):
        super().__init__(population)
        self._list = lst

    def log(self, data):
        self._list.extend(data)

    # TODO Implement this
    def getPrettyTable(self):
        raise NotImplementedError()


class StreamLogger(Logger):

    def __init__(self, population, stream, heading, dialect, precision):
        # stream is a "file object" in parsletongue
        super().__init__(population)
        self._stream = stream
        self._heading = heading
        self._dialect = dialect
        self._floatFormat = "{{0:.{}f}}".format(precision)
        self._writer = None

    def log(self, data):
        if not self._writer:
            self._writer = csv.writer(self._stream, self._dialect)
        if not self._started:
            if self._heading:
                self._writer.writerow(
                    tuple(c for c in logDetails
                          if c in self._population._logging and c in logIndecies))
            self._started = True
        for row in data:
            self._writer.writerow([self.formatValue(x) for x in row])

    def formatValue(self, v):
        if v is None:
            return ""
        elif isinstance(v, int):
            return v
        elif isinstance(v, float):
            return self._floatFormat.format(v)
        else:
            try:
                return str(v)
            except:
                return "<unprintable>"


    def flush(self):
        if self._stream:
            self._stream.flush()

    # Note that closing a StreamLogger does not close the stream as
    # that could be rude to our caller, which has the responsibiliy of
    # closing any stream it opened.
    def close(self):
        super().close()
        self._writer = None


class FileLogger(StreamLogger):

    def __init__(self, population, filename, heading, dialect, precision):
        super().__init__(population, None, heading, dialect, precision)
        self._filename = filename

    def log(self, data):
        if not self._stream:
            self._stream = open(self._filename,
                                "a" if self._started else "w",
                                newline='')
        super().log(data)

    def close(self):
        super().close()
        if self._stream:
            self._stream.close()
            self._stream = None

    def getPrettyTable(self):
        if not self._heading:
            # TODO fix this
            raise IllegalStateError("Can't yet display a log without headings")
        with open(self._filename, "r") as f:
            return from_csv(f)


class DatabaseLogger(Logger):

    def __init__(self, population, database, table, create, clear):
        super().__init__(population)
        if hasattr(database, "cursor"):
            # a connection, or connection-like object
            self._database = None
            self._connection = database
        else:
            # a database name
            self._database = str(database)
            self._connection = None
        self._table = table
        self._create = create
        self._clear = clear
        self._pending = []

    def log(self, data):
        self._pending.extend([[self.formatValue(x) for x in row] for row in data])
        if len(self._pending) >= DB_BUFFER_LENGTH:
            self.flush()

    def formatValue(self, v):
        if v is None or isinstance(v, int) or isinstance(v, float):
            return v
        else:
            try:
                return str(v)
            except:
                return "<unprintable>"

    def flush(self):
        if not self._pending:
            return
        if not self._connection:
            self._connection = sqlite3.connect(self._database)
        if not self._cursor:
            self._cursor = self._connection.cursor()
        if not self._started:
            self._insertStmt = "INSERT INTO {} VALUES({})".format(
                self._table,
                ",".join(repeat('?', sum(1 for x in self._population._logging
                                         if x in logIndecies))))
            if self._create:
                self._cursor.execute("CREATE TABLE IF NOT EXISTS {} ({})".format(
                    self._table,
                    ",".join("{} {}".format(d, logTypes[d])
                             for d in logDetails
                             if d in self._population._logging and d in logIndecies)))
                self._connection.commit()
            if self._clear:
                self._cursor.execute("DELETE from '{}'".format(self._table))
                self._connection.commit()
        self._cursor.executemany(self._insertStmt, self._pending)
        self._connection.commit()
        self._pending.clear()

    def close(self):
        super().close()
        if self._database:
            if self._connection:
                self._connection.close()
            self._cursor = None

    # TODO Implement this
    def getPrettyTable(self):
        raise NotImplementedError()


class PyIBLException(Exception):
    pass

class PyIBLWarning(UserWarning, PyIBLException):
    pass

def pyiblwarn(message):
    warn(message, PyIBLWarning, 2)

class IllegalArgumentError(ValueError, PyIBLException):
    """Raised by many PyIBL methods when passed a somehow defective argument; inherits from :exc:`ValueError`."""
    pass

class IllegalStateError(RuntimeError, PyIBLException):
    """Raised by many PyIBL methods when called in a way that exposes some incorrect internal state; inherits from :exc:`RuntimeError`."""
    pass

def isHashable(value):
    try:
        hash(value)
    except:
        return False
    return True

def ensurePossibleDecision(value):
    try:
        if value is None:
            raise TypeError()
        hash(value)             # raises a TypeError if not hashable
    except TypeError:
        raise IllegalArgumentError("{} cannot be used as a decision".format(value))

def ensureAttributeValue(value):
    try:
        hash(value)             # raises a TypeError if not hashable
    except TypeError:
        raise IllegalArgumentError("{} cannot be used as an attribute value".format(value))

def ensurePossibleOutcome(value):
    try:
        value <= 1                   # value doesn't matter, just ensuring comparable
    except TypeError as err:
        raise IllegalArgumentError(
            "outcome {} does not have a (non-complex) numeric value ({})".format(
                value, err))

def requirePyIBLVersion(minimum, maximum=None, description=None):
    # minimum is inclusive; maximum is exclusive, and should not include
    # things like 'a1' or '.dev2'
    normalized = NormalizedVersion(__version__)
    minimum = NormalizedVersion(minimum)
    if maximum:
        maximum = NormalizedVersion("{}a0.dev0".format(maximum))
    if normalized >= minimum and (not maximum or normalized < maximum):
        return
    msg = ("{} cannot use this version ({}) of PyIBL; it requires at least " +
           "PyIBL version {}").format(
               description or "The application or library using PyIBL",
               str(normalized),
               str(minimum))
    if not maximum:
        msg += "."
    else:
        msg += ", and will not run in PyIBL version {} or later.".format(str(maximum))
    raise IllegalStateError(msg)



# Local variables:
# fill-column: 90
# End:
