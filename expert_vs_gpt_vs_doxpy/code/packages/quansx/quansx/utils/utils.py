import itertools
import math

def flatten(it, as_list=False):
	flattened_it = (
		y
		for x in it
		for y in x
	)
	# flattened_it = itertools.chain.from_iterable(it)
	if as_list:
		flattened_it = list(flattened_it)
	return flattened_it

def get_chunks(it, elements_per_chunk=None, number_of_chunks=None):
	"""Divide a list of nodes `it` in `n` chunks"""
	# assert elements_per_chunk or number_of_chunks
	# it = iter(it)
	if not isinstance(it, (list,tuple)):
		it = tuple(it)
	if elements_per_chunk:
		return (
			it[i*elements_per_chunk:(i+1)*elements_per_chunk]
			for i in range(math.ceil(len(it)/elements_per_chunk))
		)
		# return iter(lambda: tuple(itertools.islice(it, elements_per_chunk)), ())
	return (
		it[i::number_of_chunks]
		for i in range(number_of_chunks)
	)
	# return (tuple(itertools.islice(it, i, None, number_of_chunks)) for i in range(number_of_chunks))
