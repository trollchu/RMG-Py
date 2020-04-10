#!/usr/bin/env python3
# encoding: utf-8

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2019 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
A module for generating ESS adapters
"""

import logging
import os
from typing import Type

from .adapter import ESSAdapter

from rmgpy.exceptions import InputError
from rmgpy.qm.qmdata import QMData
from rmgpy.qm.symmetry import PointGroupCalculator

_registered_ess_adapters = {}


def register_ess_adapter(ess: str,
                         ess_class: Type[ESSAdapter],
                         ) -> None:
    """
    A register for the ESS adapters.

    Args:
        ess: A string representation for an ESS adapter
        ess_class: The ESS adapter class

    Raises:
        TypeError: If ``ess_class`` is not an ``ESSAdapter`` instance.
    """
    if not issubclass(ess_class, ESSAdapter):
        raise TypeError(f'{ess_class} is not an ESSAdapter')
    _registered_ess_adapters[ess] = ess_class


def ess_factory(fullpath: str) -> Type[ESSAdapter]:
    """
    A factory generating the ESS adapter corresponding to ``ess_adapter``.
    Given a path to the log file of a QM software, determine whether it is
    Gaussian, Molpro, QChem, Orca, or TeraChem

    Args:
        fullpath (str): The disk location of the output file of interest.

    Returns:
        Type[ESSAdapter]: The requested ESSAdapter child, initialized with the respective arguments.
    """

    ess_name = None
    if os.path.splitext(fullpath)[1] in ['.xyz', '.dat', '.geometry']:
        ess_name = 'terachem'
    else:
        with open(fullpath, 'r') as f:
            line = f.readline()
            while ess_name is None and line != '':
                if 'gaussian' in line.lower():
                    ess_name = 'gaussian'
                    break
                elif 'molpro' in line.lower():
                    ess_name = 'molpro'
                    break
                elif 'O   R   C   A' in line or 'orca' in line.lower():
                    ess_name = 'orca'
                    break
                elif 'qchem' in line.lower():
                    ess_name = 'qchem'
                    break
                elif 'terachem' in line.lower():
                    ess_name = 'terachem'
                    break
                line = f.readline()
    if ess_name is None:
        raise InputError(f'The file at {fullpath} could not be identified as a '
                         f'Gaussian, Molpro, Orca, QChem, or TeraChem log file.')

    return _registered_ess_adapters[ess_name](path=fullpath)


class Log(object):
    def __init__(self, fullpath: str):
        """
        This class is not a part of the Factory/Adapter design class. It was added here to keep Arkane
        backward compatible with current input files that use ``Log`` calls. Added a .ess attribute that stores
        the specific ESS Adapter, along with shortcut methods to call the respective Adapter method.
        """
        self._path = fullpath
        self.ess = None
        self.set_ess()

    @property
    def path(self):
        """
        The path to an ESS log file
        """
        return self._path

    @path.setter
    def path(self, value):
        """
        Allow setting the ESS log file path, and self.ess which is derived from the log file.
        """
        self._path = value
        self.set_ess()

    def set_ess(self):
        """
        Set self.ess according to self._path if the latter is a file
        """
        if os.path.isfile(self.path):
            self.ess = ess_factory(self.path)

    def get_number_of_atoms(self):
        """
        Return the number of atoms in the molecular configuration.
        """
        return self.ess.get_number_of_atoms()

    def load_force_constant_matrix(self):
        """
        Return the force constant matrix (in Cartesian coordinates).
        """
        return self.ess.load_force_constant_matrix()

    def load_geometry(self):
        """
        Return the optimum geometry of the molecular configuration.
        """
        return self.ess.load_geometry()

    def load_conformer(self, symmetry=None, spin_multiplicity=0, optical_isomers=None, label=''):
        """
        Load the molecular degree of freedom data from a frequency calculation.
        """
        return self.ess.load_conformer(symmetry=symmetry, spin_multiplicity=spin_multiplicity,
                                       optical_isomers=optical_isomers, label=label)

    def load_energy(self, zpe_scale_factor=1.):
        """
        Load the energy.
        """
        return self.ess.load_energy(zpe_scale_factor=zpe_scale_factor)

    def load_zero_point_energy(self):
        """
        Load the unscaled zero-point energy.
        """
        return self.ess.load_zero_point_energy()

    def load_scan_energies(self):
        """
        Extract the optimized energies from a potential energy scan.
        """
        return self.ess.load_scan_energies()

    def load_scan_pivot_atoms(self):
        """
        Extract the atom numbers which the rotor scan pivots around.
        """
        return self.ess.load_scan_pivot_atoms()

    def load_scan_frozen_atoms(self):
        """
        Extract the atom numbers where were frozen during the scan.
        """
        return self.ess.load_scan_frozen_atoms()

    def load_negative_frequency(self):
        """
        Return the negative frequency from a transition state frequency calculation.
        """
        return self.ess.load_negative_frequency()

    def get_symmetry_properties(self):
        """
        This method uses the symmetry package from RMG's QM module
        and returns a tuple where the first element is the number
        of optical isomers, the second element is the symmetry number,
        and the third element is the point group identified.
        """
        coordinates, atom_numbers, _ = self.load_geometry()
        unique_id = '0'  # Just some name that the SYMMETRY code gives to one of its jobs
        # Scratch directory that the SYMMETRY code writes its files in:
        scr_dir = os.path.join(os.path.abspath('.'), str('scratch'))
        if not os.path.exists(scr_dir):
            os.makedirs(scr_dir)
        try:
            qmdata = QMData(
                groundStateDegeneracy=1,  # Only needed to check if valid QMData
                numberOfAtoms=len(atom_numbers),
                atomicNumbers=atom_numbers,
                atomCoords=(coordinates, str('angstrom')),
                energy=(0.0, str('kcal/mol'))  # Only needed to avoid error
            )
            # Dynamically create custom class to store the settings needed for the point group calculation
            # Normally, it expects an rmgpy.qm.main.QMSettings object, but we don't need all of those settings
            settings = type(str(''), (),
                            dict(symmetryPath=str('symmetry'), scratchDirectory=scr_dir))()
            pgc = PointGroupCalculator(settings, unique_id, qmdata)
            pg = pgc.calculate()
            if pg is not None:
                optical_isomers = 2 if pg.chiral else 1
                symmetry = pg.symmetry_number
                logging.debug("Symmetry algorithm found {0} optical isomers and a symmetry number of {1}".format(
                    optical_isomers, symmetry))
            else:
                logging.error('Symmetry algorithm errored when computing point group\nfor log file located at{0}.\n'
                              'Manually provide values in Arkane input.'.format(self.ess.path))
            return optical_isomers, symmetry, pg.point_group
        finally:
            shutil.rmtree(scr_dir)

    def get_D1_diagnostic(self):
        """
        Returns the D1 diagnostic from the output log for certain quantum jobs.
        """
        return self.ess.get_D1_diagnostic()

    def get_T1_diagnostic(self):
        """
        Returns the T1 diagnostic from the output log for certain quantum jobs.
        """
        return self.ess.get_T1_diagnostic()
