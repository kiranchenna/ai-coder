```javascript
const express = require('express');
const { createConnection } = require('typeorm');
const authRoutes = require('./api/auth');
const taskRoutes = require('./api/tasks');

const app = express();
app.use(express.json());

createConnection({
  type: 'postgres',
  url: process.env.DATABASE_URL,
  entities: [__dirname + '/models/*.js'],
  synchronize: true,
}).then(() => {
  console.log('Database connected');
});

app.use('/api/auth', authRoutes);
app.use('/api/tasks', taskRoutes);

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```
