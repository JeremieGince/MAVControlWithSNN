from typing import Callable, Dict, List, NamedTuple, Optional

from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from PythonAcademy.src.buffers import ReplayBuffer
from PythonAcademy.src.utils import TrainingHistory


class CompletionCriteria(NamedTuple):
	"""
	Completion criteria for a lesson.
	"""
	measure: str
	min_lesson_length: int
	threshold: float

	@staticmethod
	def default_criteria() -> 'CompletionCriteria':
		return CompletionCriteria(measure='Rewards', min_lesson_length=1, threshold=0.9)


class Lesson:
	def __init__(
			self,
			name,
			channel: EnvironmentParametersChannel,
			params: Dict[str, float],
			completion_criteria: CompletionCriteria = CompletionCriteria.default_criteria(),
			teacher=None
	):
		assert teacher is None or hasattr(teacher, 'buffer'), 'Teacher must have a replay buffer.'
		self.name = name
		self.completion_criteria = completion_criteria
		self.channel = channel
		self.params = params
		self._num_iterations = 0
		self._teacher = teacher
		self._result = None

	@property
	def teacher(self):
		return self._teacher

	@property
	def teacher_buffer(self) -> Optional[ReplayBuffer]:
		"""
		Returns the replay buffer for the lesson.
		"""
		if self._teacher is not None:
			buffer = self._teacher.buffer
			assert isinstance(buffer, ReplayBuffer), 'Teacher must have a replay buffer.'
			return self._teacher.buffer
		return None

	@property
	def is_completed(self):
		"""
		Returns True if the lesson is completed, False otherwise.
		"""
		return self._result is not None

	def set_result(self, result):
		self._result = result
		self._teacher.discard_buffer()

	def start(self):
		"""
		Starts the lesson.
		"""
		for key, value in self.params.items():
			self.channel.set_float_parameter(key, value)

	def on_iteration_end(self, training_history: TrainingHistory):
		"""
		Called when an iteration ends.
		"""
		self._num_iterations += 1

	def check_completion_criteria(self, training_history: TrainingHistory) -> bool:
		"""
		Checks if the lesson is completed.
		"""
		if self.completion_criteria.min_lesson_length > self._num_iterations:
			return False
		result = training_history[self.completion_criteria.measure][-1]
		if result >= self.completion_criteria.threshold:
			self.set_result(result)
		return self.is_completed


class Curriculum:
	def __init__(self, name: str = "Curriculum", description: str = "", lessons: List[Lesson] = None):
		self.name = name
		self.description = description
		self._lessons = [] if lessons is None else lessons
		self._current_lesson_idx = 0

	@property
	def map_repr(self) -> Dict[str, str]:
		progress = f"{100 * (self._current_lesson_idx + 1) / len(self._lessons):.1f} %"
		return {self.name: f'(lesson: {self._lessons[self._current_lesson_idx].name}) {progress}'}

	def add_lesson(self, lesson: Lesson):
		self._lessons.append(lesson)

	def __str__(self):
		map_repr = self.map_repr
		return f'{self.name} {map_repr[self.name]}'

	def __repr__(self):
		return str(self)

	@property
	def teacher_buffer(self) -> Optional[ReplayBuffer]:
		"""
		Returns the current teacher buffer.
		"""
		return self._lessons[self._current_lesson_idx].teacher_buffer

	def on_iteration_end(self, training_history: TrainingHistory) -> Dict[str, str]:
		"""
		Called when an iteration ends.
		"""
		if self._current_lesson_idx < len(self._lessons):
			lesson = self._lessons[self._current_lesson_idx]
			lesson.on_iteration_end(training_history)
			if lesson.is_completed:
				self._current_lesson_idx += 1
				lesson.start()
		return self.map_repr


