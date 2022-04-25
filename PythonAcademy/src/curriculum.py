from typing import Callable, List, NamedTuple

from PythonAcademy.src.utils import TrainingHistory


class CompletionCriteria(NamedTuple):
	"""
	Completion criteria for a lesson.
	"""
	measure: str
	min_lesson_length: int
	threshold: float


class Lesson:
	def __init__(self, name, completion_criteria: CompletionCriteria, teacher: Callable = None):
		self.name = name
		self.completion_criteria = completion_criteria
		self._num_iterations = 0
		self._teacher = teacher

	def start(self):
		"""
		Starts the lesson.
		"""
		raise NotImplementedError()

	def on_iteration_end(self, training_history: TrainingHistory):
		"""
		Called when an iteration ends.
		"""
		self._num_iterations += 1

	def is_completed(self, training_history: TrainingHistory):
		"""
		Returns True if the lesson is completed, False otherwise.
		"""
		raise NotImplementedError()


class Curriculum:
	def __init__(self, name, description: str = "", lessons: List[Lesson] = None):
		self.name = name
		self.description = description
		self._lessons = [] if lessons is None else lessons

	def add_lesson(self, lesson: Lesson):
		self._lessons.append(lesson)

	def __str__(self):
		return f'{self.name} ({len(self._lessons)} lessons)'

	def __repr__(self):
		return f'{self.name} ({len(self._lessons)} lessons)'



