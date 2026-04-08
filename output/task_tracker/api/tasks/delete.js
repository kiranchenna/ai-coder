```javascript
const { Task } = require('../../models/Task');

exports.delete = async (req, res) => {
  const { id } = req.params;

  try {
    const task = await Task.findOne(id);

    if (!task) {
      return res.status(404).json({ error: 'Task not found' });
    }

    await task.remove();

    res.json({ message: 'Task deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```
