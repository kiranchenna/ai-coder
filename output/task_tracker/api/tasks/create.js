```javascript
const { Task } = require('../../models/Task');

exports.create = async (req, res) => {
  const { title, description, category, deadline } = req.body;

  try {
    const task = new Task();
    task.title = title;
    task.description = description;
    task.category = category;
    task.deadline = new Date(deadline);
    task.user = req.user; // Assuming user is attached to the request object

    await task.save();

    res.status(201).json(task);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```
