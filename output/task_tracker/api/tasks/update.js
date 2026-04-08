```javascript
const { Task } = require('../../models/Task');

exports.update = async (req, res) => {
  const { id } = req.params;
  const { title, description, category, deadline } = req.body;

  try {
    const task = await Task.findOne(id);

    if (!task) {
      return res.status(404).json({ error: 'Task not found' });
    }

    task.title = title || task.title;
    task.description = description || task.description;
    task.category = category || task.category;
    task.deadline = deadline ? new Date(deadline) : task.deadline;

    await task.save();

    res.json(task);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```
